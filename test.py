import lightgbm as lgb
import numpy as np

X = np.random.rand(1000,10)
y = np.random.randint(0,3,1000)

ds = lgb.Dataset(X,label=y)

params = {
    "objective":"multiclass",
    "num_class":3,
    "device":"cuda"
}

lgb.train(params, ds, num_boost_round=1)
