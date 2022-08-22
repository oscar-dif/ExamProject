import pandas as pd
import seaborn as sns
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
from sklearn.model_selection import cross_val_score,GridSearchCV

sns.set_theme(style='whitegrid')
pd.set_option('display.max_columns',None)

df_train = pd.read_csv(r"C:\\Users\\oscar\\Desktop\\Data Engineer Assignments\\Examensprojekt\\Dataset\\train.csv")
df_test = pd.read_csv(r"C:\\Users\\oscar\\Desktop\\Data Engineer Assignments\\Examensprojekt\\Dataset\\test.csv")
df_total = pd.concat([df_train,df_test])

df_total.head()

df_train.shape, df_test.shape

df_total.info()

df_total.columns = df_total.columns.str.lower() 
df_train.columns = df_train.columns.str.lower()

df_total.id = df_total.id.astype('O')

df_total.describe().T

def grab_col_names(dataframe, cat_th=10, car_th=20, show_cols=False):
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    
    if show_cols:
        print(f"""Categorical Variables={cat_cols},\nNumerical Variables={num_cols},\nCategorical but Cardinal Variables={cat_but_car}""")
    
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df_total)

df_total[num_cols].hist(figsize=(25,20))
plt.show()

def dominant_val(dataframe, list_of_columns, percent=0.9, values=False):
    limit = dataframe.shape[0] * percent 
    cols_with_dominance = [col for col in list_of_columns if dataframe[col].value_counts().iloc[0] > limit]
    if values:
        dominant_values = [dataframe[col].value_counts().index[0] for col in cols_with_dominance]
        return zip(cols_with_dominance,dominant_values)
    return cols_with_dominance

most_zero_cols = dominant_val(df_total, num_cols, percent=0.8)
most_zero_cols

def make_it_binary(dataframe, list_of_columns, value):
    return dataframe[list_of_columns].applymap(lambda x: True if x==value else False)

df_total[most_zero_cols] = make_it_binary(df_total,most_zero_cols,0)
num_cols = [col for col in num_cols if col not in most_zero_cols]
num_cols.remove('saleprice')
len(num_cols)

plt.figure(figsize= (25,25))
for i,col in enumerate(num_cols):
    plt.subplot(5,4,i+1)
    sns.scatterplot(x=col, y="saleprice", data=df_train)

df_total.mssubclass = df_total.mssubclass.astype('category')

drop_list = ['bsmtunfsf','lotarea','lotfrontage']

corrmat = df_train.corr()
f,ax =plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True);

k = 10 #number of variables for heatmap
cols =corrmat.nlargest(k, 'saleprice')['saleprice'].index
cm =np.corrcoef(df_train[cols].values.T)
sns.set(font_scale=1.25)
hm =sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()

drop_list += ['totrmsabvgrd','garagearea']

len(cat_cols)

plt.figure(figsize= (30,60))
for i,col in enumerate(cat_cols):
    plt.subplot(13,4,i+1)
    sns.barplot(x=col, y="saleprice", data=df_train)

plt.figure(figsize= (30,60))
for i,col in enumerate(cat_cols):
    plt.subplot(13,4,i+1)
    sns.boxplot(x=col, y="saleprice", data=df_train)

df_total.kitchenabvgr.value_counts()

dominant_zip = dominant_val(df_total,cat_cols,values=True)
dominant_list = list(dominant_zip)

for col,value in dominant_list:
    print(col,value)

for col,value in dominant_list:
    df_total[col] = df_total[col].apply(lambda x: True if x==value else False)

corrmat = df_total[cat_cols].apply(lambda x : pd.factorize(x)[0]).corr(method='pearson', min_periods=1).abs()

s = pd.DataFrame(corrmat.unstack(),columns=['correlation'])

so = s.sort_values(by='correlation',kind="quicksort",ascending=False)

so[so['correlation']<1].head(10)

drop_list += ['poolqc','exterior2nd','fireplacequ']

cat_but_car

drop_list += cat_but_car

def missing_values_table(dataframe, printer=False, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df_train = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    
    if printer:
        print(missing_df_train, end="\n")
        
    if na_name:
        return na_columns

df_total.drop(drop_list,inplace=True,axis=1)
missing_values_table(df_total,printer=True)

no_cols = ["alley","bsmtqual","bsmtcond","bsmtexposure","bsmtfintype1","bsmtfintype2",
           "garagetype","garagefinish","garagequal","garagecond","fence","miscfeature"]

for col in no_cols:
    df_total[col].fillna("No",inplace=True)

missing_columns = missing_values_table(df_total,na_name=True)

missing_columns.remove('saleprice')

for col in missing_columns:
    df_total[col].fillna(df_total[col].mode()[0],inplace=True)

missing_values_table(df_total,printer=True)

cat_cols,  num_cols, cat_but_car = grab_col_names(df_total)

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df_total = one_hot_encoder(df_total,cat_cols,drop_first=True)

y_train = df_total.loc[~df_total.saleprice.isna(),'saleprice']

y_train = np.log1p(y_train)

X_train = df_total.iloc[:1460].copy()
X_train.drop('saleprice',axis=1,inplace=True)
y_train.max()

models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

lgbm_model = LGBMRegressor(random_state=46)

rmse = np.mean(np.sqrt(-cross_val_score(lgbm_model,
                                        X_train, y_train, cv=5, scoring="neg_mean_squared_error")))

rmse

lgbm_params = {"learning_rate": [0.01, 0.1],
               "n_estimators": [500, 1500]
               #"colsample_bytree": [0.5, 0.7, 1]
             }

lgbm_gs_best = GridSearchCV(lgbm_model,
                            lgbm_params,
                            cv=3,
                            n_jobs=-1,
                            verbose=True).fit(X_train, y_train)

final_model = lgbm_model.set_params(**lgbm_gs_best.best_params_).fit(X_train,y_train)

rmse = np.mean(np.sqrt(-cross_val_score(final_model, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))

rmse

X_test = df_total.iloc[1460:].drop('saleprice',axis=1).copy()

y_pred = final_model.predict(X_test)

y_pred= np.expm1(y_pred)

y_pred

pd.DataFrame({'Id':df_test.Id,'SalePrice': y_pred}).to_csv('submission.csv',index=False)
