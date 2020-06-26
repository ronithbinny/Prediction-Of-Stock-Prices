import numpy as np
import pandas as pd

dataset = pd.read_csv("something.csv")
X = dataset.iloc[:-2,1:-2].values
Y = dataset.iloc[:-2,-2].values
x = dataset.iloc[:-2,1:-2].values
y = dataset.iloc[:-2,-1].values

# Open, High, Low, Close, Adj Close, Volume
X_test = np.array([96.10,97.20,93.80,94.30,94.30,9230752]).reshape(1,-1)

#X_test = np.array([106.199997,109,106.199997,107.650002,107.650002,14532478]).reshape(1,-1)

from sklearn.linear_model import LinearRegression
regressor1 = LinearRegression()
regressor1.fit(X, Y)
high1 = regressor1.predict(X_test)

regressor11 = LinearRegression()
regressor11.fit(x, y)
low1 = regressor11.predict(X_test)


from sklearn.svm import SVR
regressor2 = SVR(kernel = "linear")
regressor2.fit(X, Y)
high2 = regressor2.predict(X_test)

regressor22 = SVR()
regressor22.fit(x, y)
low2 = regressor22.predict(X_test)


from sklearn.tree import DecisionTreeRegressor
regressor3 = DecisionTreeRegressor()
regressor3.fit(X, Y)
high3 = regressor3.predict(X_test)

regressor33 = DecisionTreeRegressor()
regressor33.fit(x, y)
low3 = regressor33.predict(X_test)


from sklearn.ensemble import RandomForestRegressor
regressor4 = RandomForestRegressor()
regressor4.fit(X, Y)
high4 = regressor4.predict(X_test)

regressor44 = RandomForestRegressor()
regressor44.fit(x, y)
low4 = regressor44.predict(X_test)



from catboost import CatBoostRegressor
regressor5 = CatBoostRegressor()
regressor5.fit(X, Y)
high5 = regressor5.predict(X_test)

regressor55 = CatBoostRegressor()
regressor55.fit(x, y)
low5 = regressor55.predict(X_test)



from xgboost import XGBRegressor
regressor6 = XGBRegressor()
regressor6.fit(X, Y)
high6 = regressor6.predict(X_test)

regressor66 = XGBRegressor()
regressor66.fit(x, y)
low6 = regressor66.predict(X_test)



import lightgbm as lgb
d_train = lgb.Dataset(X, label = Y)
d_train1 = lgb.Dataset(x, label = y)
params = {}
params['learning_rate'] = 0.125
params['boosting_type'] = 'gbdt'
params['objective'] = 'regression'
params['metric'] = 'rmsle'
params['sub_feature'] = 0.5
params['num_leaves'] = 5
params['min_data'] = 50
params['max_depth'] = 8
regressor7 = lgb.train(params,d_train,1500)
regressor77 = lgb.train(params,d_train1,1500)

high7 = regressor7.predict(X_test)
low7 = regressor77.predict(X_test)



print("High : {} \nLow : {}".format(high1,low1))
print("High : {} \nLow : {}".format(high2,low2))
print("High : {} \nLow : {}".format(high3,low3))
print("High : {} \nLow : {}".format(high4,low4))
print("High : {} \nLow : {}".format(high5,low5))
print("High : {} \nLow : {}".format(high6,low6))
print("High : {} \nLow : {}".format(high7,low7))
