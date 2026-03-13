"""
models/trainer.py
-----------------
LightGBM training with GPU acceleration (CUDA / OpenCL).

GPU Support:
- Auto-detects NVIDIA GPU via nvidia-smi / torch.cuda
- device = "cuda"   → LightGBM CUDA backend  (fastest, NVIDIA only, LGBM >= 3.3)
- device = "gpu"    → LightGBM OpenCL backend (NVIDIA + AMD, older driver fallback)
- device = "cpu"    → Pure CPU fallback

How LightGBM GPU works:
  LightGBM parallelises histogram building on the GPU.
  Speedup is most pronounced on large datasets (> 100k rows) and
  high num_leaves. You need LightGBM compiled with GPU support:

    pip install lightgbm --config-settings=cmake.define.USE_GPU=ON
  or use setup_gpu.sh (included) for full Ubuntu CUDA install.

Features vs CPU version:
- GPUProbe: auto-detects GPU type, VRAM, driver, CUDA version
- GPU-tuned default params (larger num_leaves safe on GPU)
- Per-fold VRAM monitoring
- Device info printed at startup
- Graceful fallback: CUDA -> OpenCL -> CPU
"""

import json
import shutil
import subprocess
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from sklearn.metrics import accuracy_score, f1_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
console = Console()


# ══════════════════════════════════════════════════════════════════════════════
# GPU Probe
# ══════════════════════════════════════════════════════════════════════════════

class GPUProbe:
    """
    Detects available GPU acceleration for LightGBM.

    Probe order (fully isolated — each check is try/except):
      1. nvidia-smi  → is there an NVIDIA GPU at all?
      2. _check_lgbm_cuda()  → does the installed wheel support CUDA backend?
      3. _check_lgbm_gpu()   → does it support OpenCL backend?
      4. Fallback → CPU

    Your setup: RTX 4060 + LightGBM 4.6 OpenCL wheel
    → device_type will be "gpu" (OpenCL), which is correct and fast.
    The "cuda" backend needs a source build with -DUSE_CUDA=1.
    """

    def __init__(self):
        self.device_type: str = "cpu"
        self.gpu_name: str = "N/A"
        self.vram_mb: int = 0
        self.driver_version: str = "N/A"
        self.cuda_version: str = "N/A"
        self.lgbm_gpu_support: bool = False   # OpenCL backend works
        self.lgbm_cuda_support: bool = False  # CUDA backend works (needs source build)
        self._probe()

    def _probe(self):
        nvidia = self._check_nvidia()
        amd = self._check_amd() if not nvidia else False

        if not (nvidia or amd):
            self.device_type = "cpu"
            return

        # Test backends in order: CUDA first (faster), then OpenCL
        # Each test is fully silent — errors are caught internally
        self.lgbm_cuda_support = self._check_lgbm_cuda()
        self.lgbm_gpu_support  = self._check_lgbm_gpu()

        if self.lgbm_cuda_support:
            self.device_type = "cuda"
        elif self.lgbm_gpu_support:
            self.device_type = "gpu"
            logger.info(
                "GPU detected: using OpenCL backend (device='gpu'). "
                "CUDA backend not available in this wheel — OpenCL still gives good speedup."
            )
        else:
            self.device_type = "cpu"
            logger.warning(
                "GPU detected but LightGBM has no GPU support in this build. "
                "Run: ./setup_gpu.sh to reinstall with GPU support."
            )

    def _check_nvidia(self) -> bool:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip().split("\n")[0]
            parts = [p.strip() for p in out.split(",")]
            self.gpu_name = parts[0]
            self.vram_mb = int(parts[1])
            self.driver_version = parts[2]
        except Exception:
            return False
        try:
            nv = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=cuda_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=5,
            ).decode().strip().split("\n")[0].strip()
            self.cuda_version = nv
        except Exception:
            try:
                nv2 = subprocess.check_output(["nvcc", "--version"], stderr=subprocess.DEVNULL, timeout=5).decode()
                import re
                m = re.search(r"release (\d+\.\d+)", nv2)
                self.cuda_version = m.group(1) if m else "?"
            except Exception:
                self.cuda_version = "?"
        return True

    def _check_amd(self) -> bool:
        try:
            subprocess.check_output(["rocm-smi", "--showproductname"], stderr=subprocess.DEVNULL, timeout=5)
            self.gpu_name = "AMD (ROCm)"
            return True
        except Exception:
            return False

    def _check_lgbm_gpu(self) -> bool:
        try:
            X = np.random.rand(100, 10).astype(np.float32)
            y = np.random.randint(0, 2, 100)
            ds = lgb.Dataset(X, label=y)
            lgb.train({"device": "gpu", "objective": "binary", "verbose": -1}, ds, num_boost_round=1)
            return True
        except Exception:
            return False

    def _check_lgbm_cuda(self) -> bool:
        try:
            X = np.random.rand(100, 10).astype(np.float32)
            y = np.random.randint(0, 3, 100)
            ds = lgb.Dataset(X, label=y)
            lgb.train(
                {"device": "cuda", "objective": "multiclass", "num_class": 3, "verbose": -1},
                ds, num_boost_round=1,
            )
            return True
        except Exception:
            return False

    def get_vram_free_mb(self) -> int:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, timeout=3,
            ).decode().strip().split("\n")[0]
            return int(out.strip())
        except Exception:
            return -1

    def print_info(self):
        color = {"cuda": "green", "gpu": "yellow", "cpu": "red"}[self.device_type]
        label = {
            "cuda": "CUDA  (NVIDIA) — fastest",
            "gpu":  "OpenCL (GPU)  — fast",
            "cpu":  "CPU Only      — slow",
        }[self.device_type]

        cuda_status = "[green]✅ Available[/green]" if self.lgbm_cuda_support else "[yellow]❌ Not in wheel (needs source build)[/yellow]"
        gpu_status  = "[green]✅ Available[/green]" if self.lgbm_gpu_support  else "[red]❌ Not available[/red]"

        lines = [
            f"[bold]Active device:[/bold]  [{color}]{label}[/{color}]",
            f"[bold]GPU:[/bold]            {self.gpu_name}",
            f"[bold]VRAM:[/bold]           {self.vram_mb:,} MB" if self.vram_mb else "[bold]VRAM:[/bold]           N/A",
            f"[bold]Driver:[/bold]         {self.driver_version}",
            f"[bold]CUDA version:[/bold]   {self.cuda_version}",
            "",
            f"[bold]LGBM CUDA backend:[/bold]   {cuda_status}",
            f"[bold]LGBM OpenCL backend:[/bold] {gpu_status}",
        ]
        console.print(Panel("\n".join(lines), title="⚡ Hardware Accelerator", border_style=color))


# ══════════════════════════════════════════════════════════════════════════════
# LGBMTrainer
# ══════════════════════════════════════════════════════════════════════════════

class LGBMTrainer:
    """
    GPU-accelerated LightGBM trainer.

    Label mapping:
        -1 (short) -> class 0
         0 (hold)  -> class 1
         1 (long)  -> class 2

    config keys:
        device: None (auto) | "cuda" | "gpu" | "cpu"
    """

    LABEL_MAP = {-1: 0, 0: 1, 1: 2}
    INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}

    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.save_dir = Path(cfg.get("save_dir", "models/saved"))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cv_splits = cfg.get("cv_splits", 5)
        self.test_size = cfg.get("test_size_bars", 720)
        self.early_stopping = cfg.get("early_stopping_rounds", 50)
        self.tune_hyperparams = cfg.get("tune_hyperparams", False)
        self.device_override = cfg.get("device", None)

        # Run GPU probe
        self.gpu = GPUProbe()
        self.device = self.device_override or self.gpu.device_type

        base_params = cfg.get("params") or self._default_params(self.device)
        self.params = self._inject_gpu_params(base_params, self.device, self.gpu.vram_mb)

        self.model: lgb.Booster | None = None
        self.feature_names: list = []

    # ─── Public API ─────────────────────────────────────────────────────────

    def train(self, features: pd.DataFrame, labels: pd.Series, symbol: str, timeframe: str) -> dict:
        self.gpu.print_info()
        console.print(Panel.fit(
            f"[bold cyan]Training LightGBM[/bold cyan]\n"
            f"Symbol: {symbol} | Timeframe: {timeframe}\n"
            f"Features: {features.shape[1]} | Samples: {len(features):,}\n"
            f"Device: [bold]{self.device.upper()}[/bold]",
            title="🤖 Model Training",
        ))

        X, y = self._align_data(features, labels)
        y_enc = y.map(self.LABEL_MAP).values
        self.feature_names = list(X.columns)

        if self.tune_hyperparams:
            self.params = self._tune(X, y_enc)

        cv_metrics = self._walk_forward_cv(X, y_enc)

        logger.info(f"Training final model on full dataset (device={self.device})...")
        # Scale min_child_samples for the full dataset too
        final_mcs = min(self.params.get("min_child_samples", 50), max(1, len(X) // 100))
        self.model = self._fit(X.values, y_enc, X.values, y_enc, extra_params={"min_child_samples": final_mcs})

        model_path = self._save(symbol, timeframe, cv_metrics)
        self._print_cv_results(cv_metrics)
        self._print_feature_importance(top_n=20)

        return {
            "model_path": str(model_path),
            "cv_metrics": cv_metrics,
            "n_features": len(self.feature_names),
            "n_samples": len(X),
            "device": self.device,
        }

    def predict(self, X: pd.DataFrame) -> tuple:
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        proba = self.model.predict(X[self.feature_names].values)
        classes = np.argmax(proba, axis=1)
        labels = np.array([self.INV_LABEL_MAP[c] for c in classes])
        return labels, proba

    # ─── GPU Param Injection ─────────────────────────────────────────────────

    @staticmethod
    def _inject_gpu_params(params: dict, device: str, vram_mb: int) -> dict:
        """
        Inject GPU-specific LightGBM params.

        device_type:   "cuda" | "gpu" | "cpu"
        gpu_device_id: which GPU to use (0 = first)
        gpu_use_dp:    False = float32 (much faster than float64 on GPU)
        max_bin:       255 = finer histograms; GPU handles this cheaply
        num_leaves:    safe to increase on GPU — gated by VRAM

        RTX 4060 (8GB) → device="gpu" (OpenCL), num_leaves=255, max_bin=255
        """
        p = dict(params)
        p["device_type"] = device

        if device in ("cuda", "gpu"):
            p.setdefault("gpu_device_id", 0)
            p.setdefault("gpu_use_dp", False)   # float32 = ~2x faster on GPU
            p.setdefault("max_bin", 255)         # GPU histogram bins — 255 is cheap on GPU

            # Scale num_leaves with VRAM
            if vram_mb >= 8192:
                p.setdefault("num_leaves", 255)  # RTX 4060 8GB — full capacity
            elif vram_mb >= 4096:
                p.setdefault("num_leaves", 127)
            elif vram_mb >= 2048:
                p.setdefault("num_leaves", 63)
            # < 2GB → keep whatever default was passed in

        return p

    # ─── Training Core ───────────────────────────────────────────────────────

    # Minimum samples we need in val AND train set for a reliable fold
    MIN_VAL_SAMPLES   = 2_000   # must have at least 2k bars to evaluate on
    MIN_TRAIN_SAMPLES = 10_000  # must have at least 10k bars to train on

    def _resolve_cv_params(self, n: int) -> tuple[int, int]:
        """
        Auto-scale test_size and cv_splits based on dataset size.

        Problem: config default test_size=720 was designed for hourly data
        (720 bars = 30 days). On 1m data, 720 bars = only 12 hours — way too
        small for LightGBM to find splits on.

        Rule: val fold = max(config_value, 10% of n), capped so train >= 50%.
        cv_splits reduced if folds would be too small.
        """
        # Use whichever is larger: config value or 10% of dataset
        raw_test = max(self.test_size, int(n * 0.10))

        # Cap so training set is always at least 50% of data
        max_test = int(n * 0.40)
        test_size = min(raw_test, max_test)

        step = test_size // self.cv_splits
        if step < self.MIN_VAL_SAMPLES:
            # Reduce cv_splits so each fold is big enough
            step = self.MIN_VAL_SAMPLES
            cv_splits = max(1, test_size // step)
        else:
            cv_splits = self.cv_splits

        if self.test_size != test_size or self.cv_splits != cv_splits:
            logger.info(
                f"CV auto-scaled for {n:,} samples: "
                f"test_size {self.test_size}→{test_size} | "
                f"cv_splits {self.cv_splits}→{cv_splits} | "
                f"fold_size={test_size // max(cv_splits,1):,}"
            )
        return test_size, cv_splits

    def _safe_min_child_samples(self, n_train: int) -> int:
        """
        Scale min_child_samples so it's never larger than 1% of training set.
        Prevents the best_split_info.left_count > 0 crash.
        """
        configured = self.params.get("min_child_samples", 50)
        safe = min(configured, max(1, n_train // 100))
        if safe != configured:
            logger.info(f"min_child_samples scaled: {configured} → {safe} (train={n_train:,})")
        return safe

    def _walk_forward_cv(self, X: pd.DataFrame, y: np.ndarray) -> list:
        n = len(X)
        metrics = []

        test_size, cv_splits = self._resolve_cv_params(n)
        step = test_size // max(cv_splits, 1)

        logger.info(f"Walk-forward CV: {cv_splits} folds | step={step:,} | device={self.device}")

        for fold in range(cv_splits):
            val_end   = n - fold * step
            val_start = val_end - step
            if val_start < self.MIN_TRAIN_SAMPLES:
                logger.warning(f"Fold {fold+1}: train set too small ({val_start:,} < {self.MIN_TRAIN_SAMPLES:,}), stopping CV early")
                break

            X_tr,  y_tr  = X.iloc[:val_start],   y[:val_start]
            X_val, y_val = X.iloc[val_start:val_end], y[val_start:val_end]

            if len(X_val) < self.MIN_VAL_SAMPLES:
                logger.warning(f"Fold {fold+1}: val set too small ({len(X_val):,}), skipping")
                continue

            # Scale min_child_samples to training set size — prevents left_count crash
            safe_mcs = self._safe_min_child_samples(len(X_tr))

            vram_free = self.gpu.get_vram_free_mb()
            vram_str  = f" | VRAM free: {vram_free} MB" if vram_free > 0 else ""
            logger.info(f"Fold {fold+1}/{cv_splits} — train={len(X_tr):,} val={len(X_val):,} min_child={safe_mcs}{vram_str}")

            model = self._fit(X_tr.values, y_tr, X_val.values, y_val, extra_params={"min_child_samples": safe_mcs})
            proba = model.predict(X_val.values)
            preds = np.argmax(proba, axis=1)

            fm = {
                "fold":        fold + 1,
                "train_size":  len(X_tr),
                "val_size":    len(X_val),
                "accuracy":    float(accuracy_score(y_val, preds)),
                "f1_macro":    float(f1_score(y_val, preds, average="macro",    zero_division=0)),
                "f1_weighted": float(f1_score(y_val, preds, average="weighted", zero_division=0)),
                "device":      self.device,
            }
            metrics.append(fm)
            logger.info(f"  acc={fm['accuracy']:.4f}  f1={fm['f1_macro']:.4f}")

        return metrics

    # Keywords that indicate a GPU-side crash (even if the message looks "generic")
    _GPU_ERROR_HINTS = (
        "cuda", "gpu", "opencl",
        # This one fires on OpenCL backend when tree building fails internally —
        # the message looks like a generic assert but it's a GPU codepath bug:
        "best_split_info.left_count",
        "check failed",          # catches other internal GPU assert failures
        "treelearner",           # all tree-learner crashes on GPU backend
    )

    def _cpu_params(self, params: dict) -> dict:
        """Strip all GPU keys and force device_type=cpu."""
        drop = {"device_type", "gpu_device_id", "gpu_use_dp", "max_bin"}
        p = {k: v for k, v in params.items() if k not in drop}
        p["device_type"] = "cpu"
        # Restore conservative max_bin for CPU (255 can be slow on CPU)
        p["max_bin"] = 63
        return p

    def _fit(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray,
             extra_params: dict | None = None) -> lgb.Booster:

        params = {**self.params, "objective": "multiclass", "num_class": 3, **(extra_params or {})}

        # CUDA backend requires float32 input
        if self.device == "cuda":
            X_train = X_train.astype(np.float32)
            X_val   = X_val.astype(np.float32)

        train_ds = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names, free_raw_data=True)
        val_ds   = lgb.Dataset(X_val,   label=y_val,   reference=train_ds,              free_raw_data=True)

        callbacks = [
            lgb.early_stopping(self.early_stopping, verbose=False),
            lgb.log_evaluation(period=-1),
        ]

        # ── First attempt (GPU if available) ────────────────────────────────
        if self.device in ("cuda", "gpu"):
            try:
                return lgb.train(params, train_ds, valid_sets=[val_ds], callbacks=callbacks)
            except lgb.basic.LightGBMError as e:
                err = str(e).lower()
                if any(hint in err for hint in self._GPU_ERROR_HINTS):
                    logger.warning(
                        f"GPU training failed ({type(e).__name__}: {str(e)[:120]})"
                        f"→ Falling back to CPU for this fold."
                    )
                    # Rebuild datasets — they may have been consumed
                    train_ds = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names, free_raw_data=True)
                    val_ds   = lgb.Dataset(X_val,   label=y_val,   reference=train_ds,              free_raw_data=True)
                    params   = self._cpu_params(params)
                    self.device = "cpu"
                    logger.info("Switched to CPU permanently for remaining folds.")
                else:
                    raise  # non-GPU error — let it bubble up

        # ── CPU path (either originally CPU, or fell back from GPU) ─────────
        return lgb.train(params, train_ds, valid_sets=[val_ds], callbacks=callbacks)

    def _align_data(self, features: pd.DataFrame, labels: pd.Series):
        common = features.index.intersection(labels.index)
        X = features.loc[common].copy()
        y = labels.loc[common].copy()
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        valid = X.notna().all(axis=1) & y.notna()
        X, y = X[valid], y[valid]
        logger.info(f"Aligned dataset: {len(X):,} samples")
        return X, y

    # ─── Hyperparameter Tuning ───────────────────────────────────────────────

    def _tune(self, X: pd.DataFrame, y: np.ndarray, n_trials: int = 50) -> dict:
        logger.info(f"Optuna HPO: {n_trials} trials on {self.device}")
        n = len(X)
        split = int(n * 0.8)

        def objective(trial):
            p = {
                "objective": "multiclass",
                "num_class": 3,
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_leaves": trial.suggest_int("num_leaves", 15, 255),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-4, 10.0, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-4, 10.0, log=True),
                "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
                "verbose": -1,
            }
            p = self._inject_gpu_params(p, self.device, self.gpu.vram_mb)
            model = self._fit(X.iloc[:split].values, y[:split], X.iloc[split:].values, y[split:], extra_params=p)
            preds = np.argmax(model.predict(X.iloc[split:].values), axis=1)
            return f1_score(y[split:], preds, average="macro", zero_division=0)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True, n_jobs=1)
        best = study.best_params
        logger.info(f"Best params: {best} | F1={study.best_value:.4f}")
        return self._inject_gpu_params({**self._default_params(self.device), **best}, self.device, self.gpu.vram_mb)

    # ─── Save / Load ────────────────────────────────────────────────────────

    def _save(self, symbol: str, timeframe: str, cv_metrics: list) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        safe = symbol.replace("/", "_")
        name = f"{safe}_{timeframe}_{ts}"

        model_path = self.save_dir / f"{name}.lgb"
        meta_path = self.save_dir / f"{name}_meta.json"

        self.model.save_model(str(model_path))

        meta = {
            "symbol": symbol,
            "timeframe": timeframe,
            "trained_at": ts,
            "device": self.device,
            "gpu_name": self.gpu.gpu_name,
            "cuda_version": self.gpu.cuda_version,
            "feature_names": self.feature_names,
            "cv_metrics": cv_metrics,
            "params": self.params,
            "avg_accuracy": float(np.mean([m["accuracy"] for m in cv_metrics])),
            "avg_f1": float(np.mean([m["f1_macro"] for m in cv_metrics])),
        }
        meta_path.write_text(json.dumps(meta, indent=2))

        for suffix in [".lgb", "_meta.json"]:
            src = self.save_dir / f"{name}{suffix}"
            dst = self.save_dir / f"{safe}_{timeframe}_latest{suffix}"
            shutil.copy(src, dst)

        logger.info(f"Model saved -> {model_path}  (device={self.device})")
        return model_path

    def load(self, symbol: str, timeframe: str) -> "LGBMTrainer":
        safe = symbol.replace("/", "_")
        model_path = self.save_dir / f"{safe}_{timeframe}_latest.lgb"
        meta_path = self.save_dir / f"{safe}_{timeframe}_latest_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(f"No model at {model_path}. Run: python main.py train")

        self.model = lgb.Booster(model_file=str(model_path))
        meta = json.loads(meta_path.read_text())
        self.feature_names = meta["feature_names"]
        trained_device = meta.get("device", "cpu")

        logger.info(
            f"Loaded {symbol} model | trained_on={trained_device} | "
            f"acc={meta['avg_accuracy']:.3f} f1={meta['avg_f1']:.3f}"
        )
        return self

    # ─── Display ────────────────────────────────────────────────────────────

    def _print_cv_results(self, cv_metrics: list):
        table = Table(title="Walk-Forward CV Results", header_style="bold blue")
        for col in ["Fold", "Train Bars", "Val Bars", "Accuracy", "F1 Macro", "F1 Weighted", "Device"]:
            table.add_column(col, justify="right")

        for m in cv_metrics:
            dev = m.get("device", "cpu")
            color = "green" if dev == "cuda" else "yellow" if dev == "gpu" else "dim"
            table.add_row(
                str(m["fold"]),
                f"{m['train_size']:,}",
                f"{m['val_size']:,}",
                f"{m['accuracy']:.4f}",
                f"{m['f1_macro']:.4f}",
                f"{m['f1_weighted']:.4f}",
                f"[{color}]{dev}[/{color}]",
            )

        avg_acc = np.mean([m["accuracy"] for m in cv_metrics])
        avg_f1 = np.mean([m["f1_macro"] for m in cv_metrics])
        table.add_row("[bold]AVG[/bold]", "", "", f"[bold]{avg_acc:.4f}[/bold]", f"[bold]{avg_f1:.4f}[/bold]", "", "")
        console.print(table)

    def _print_feature_importance(self, top_n: int = 20):
        if self.model is None:
            return
        imp = pd.Series(
            self.model.feature_importance(importance_type="gain"),
            index=self.feature_names,
        ).sort_values(ascending=False)

        table = Table(title=f"Top {top_n} Features (Gain)", header_style="bold magenta")
        table.add_column("Rank", justify="right")
        table.add_column("Feature")
        table.add_column("Gain", justify="right")

        for i, (feat, val) in enumerate(imp.head(top_n).items()):
            table.add_row(str(i + 1), feat, f"{val:,.1f}")

        console.print(table)

    @staticmethod
    def _default_params(device: str = "cpu") -> dict:
        """
        GPU default params differ from CPU:
        - num_leaves: GPU can handle 127-255 (CPU: keep at 63)
        - n_estimators: higher on GPU since each round is faster
        - max_bin: 255 always recommended for GPU
        """
        return {
            "objective": "multiclass",
            "num_class": 3,
            "n_estimators": 500 if device == "cpu" else 1000,
            "learning_rate": 0.05,
            "max_depth": -1,
            "num_leaves": 63 if device == "cpu" else 127,
            "min_child_samples": 50,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "max_bin": 255,
            "verbose": -1,
        }