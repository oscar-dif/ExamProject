#import libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.model_selection import train_test_split

# load data

df = pd.read_csv('transformed.csv')

#create categories
cat_thier=16
car_thier=20
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]
cat_but_ord = ['lotshape', 'landslope', 'exterqual', 'extercond', 'bsmtqual', 
                'bsmtcond', 'bsmtexposure', 'bsmtfintype1', 'bsmtfintype2', 'heatingqc', 
                'kitchenqual', 'functional', 'fireplacequ', 'garagefinish', 'garagequal', 
                'garagecond', 'poolqc', 'fence', 'overallqual', 'overallcond']

cat_but_car = [col for col in df.columns if df[col].nunique() > car_thier and
                   df[col].dtypes == "O"]
num_but_cat = [col for col in df.columns if df[col].nunique() <= cat_thier and
                   df[col].dtypes != "O" and col not in cat_but_ord]

cat_cols = [col for col in cat_cols if col not in cat_but_car and cat_but_ord]
cat_cols = cat_cols + num_but_cat

# count and list the created groups.
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
num_cols = [col for col in num_cols if col not in num_but_cat and col not in cat_but_ord]

# creat a new target variable

df['price_per_sqft'] = df['saleprice']/ df['grlivarea']

#featurer engineering

df= df.replace({"bsmtcond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
                "bsmtexposure" : {"None" : 0, "No": 1, "Mn" : 2, "Av": 3, "Gd" : 4},
                "bsmtfintype1" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                "bsmtfintype2" : {"None" : 0, "Unf" : 1, "LwQ": 2, "Rec" : 3, "BLQ" : 4, "ALQ" : 5, "GLQ" : 6},
                "bsmtqual" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA": 3, "Gd" : 4, "Ex" : 5},
                "extercond" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
              "exterqual" : {"Po" : 1, "Fa" : 2, "TA": 3, "Gd": 4, "Ex" : 5},
              "fireplacequ" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
              "functional" : {"Sal" : 1, "Sev" : 2, "Maj2" : 3, "Maj1" : 4, "Mod": 5, "Min2" : 6, "Min1" : 7, "Typ" : 8},
              "garagecond" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
              "garagefinish" : {"None" : 0, "Unf" : 1, "RFn" : 2, "Fin" : 3},
              "garagequal" : {"None" : 0, "Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
              "heatingqc" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
              "kitchenqual" : {"Po" : 1, "Fa" : 2, "TA" : 3, "Gd" : 4, "Ex" : 5},
              "landslope" : {"Sev" : 1, "Mod" : 2, "Gtl" : 3},
              "lotshape" : {"IR3" : 1, "IR2" : 2, "IR1" : 3, "Reg" : 4},
              "poolqc" : {"None" : 0, "Fa" : 1, "TA" : 2, "Gd" : 3, "Ex" : 4},
              "fence" : {"None":0, "MnWw": 1, "GdWo": 2, "MnPrv": 3, "GdPrv":4},
              })
#select variables
flask_variables = ['price_per_sqft', 'yearbuilt', 'yearremodadd', 'overallqual', 'exterqual', 'garagecars', 'kitchenqual', 'bsmtqual']

df = df[flask_variables]

y = df['price_per_sqft']
X = df.drop(['price_per_sqft'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
#print(X_train.shape, X_test.shape)
#print(y_train.shape, y_test.shape)

gb = GradientBoostingRegressor()
#params_gb = {
#   'loss' : ('squared_error', 'absolute_error','huber'),
#   'learning_rate' : (1.0, 0.1, 0.01),
#   'n_estimators' : (100, 200, 300)
#}

#mod_gb = GridSearchCV(gb, params_gb, cv=10)
#mod_gb.fit(X_train, y_train)
#print('Best_hyperparameter : ', mod_gb.best_params_)

#pred_gb_train = mod_gb.predict(X_train)
#print(f'RMSE train: {mean_squared_error(y_train,pred_gb_train, squared=False)}')

#pred_gb_test = mod_gb.predict(X_test)
#print(f'RMSE test: {mean_squared_error(y_test,pred_gb_test, squared=False)}')

def forecast_price(X):
    gb_best=  {'learning_rate': 0.1, 'loss': 'absolute_error', 'n_estimators': 100}
    final_model = gb.set_params(**gb_best).fit(X_train,y_train)
    return final_model.predict(X)
x1 = [2000, 2003, 7, 7,1,1,0]
x2 = {
    'yearbuilt': 2000,
    'yearremodadd': 2010,
    'overallqual': 7,
    'exterqual': 7,
    'garagecars':1,
    'kitchenqual':7,
    'bsmtqual':0
}
x2 = pd.DataFrame(x2, index=[0])
print(forecast_price(x2))


