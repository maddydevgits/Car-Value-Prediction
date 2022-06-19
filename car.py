import numpy as np
import pandas as pd
data = pd.read_csv(r'car.csv')
print(data.head())
from flask import Flask, render_template, request
import pickle
import requests
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
import seaborn as sns


data = data.drop(['Car_Name'], axis=1) 
data['current_year'] = 2020
data['no_year'] = data['current_year'] - data['Year'] 
data = data.drop(['Year','current_year'],axis = 1) 
data = pd.get_dummies(data,drop_first=True) 
data = data[['Selling_Price','Present_Price','Kms_Driven','no_year','Owner','Fuel_Type_Diesel','Fuel_Type_Petrol',
'Seller_Type_Individual','Transmission_Manual']]
print(data)

print(data.corr())

x = data.iloc[:,1:]
y = data.iloc[:,0]


model = ExtraTreesRegressor()
print(model.fit(x,y))

print(model.feature_importances_)

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200,num = 12)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5,30,num = 6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf  = [1,2,5,10]

grid = {'n_estimators': n_estimators,
        'max_features': max_features,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
       'min_samples_leaf': min_samples_leaf}
print(grid)

from sklearn.model_selection import RandomizedSearchCV, train_test_split    #importing train test split module
x_train, x_test,y_train,y_test = train_test_split(x,y,random_state=0,test_size=0.2)
from sklearn.ensemble import RandomForestRegressor 
model = RandomForestRegressor() 
hyp = RandomizedSearchCV(estimator = model,param_distributions=grid,n_iter=10,scoring= 'neg_mean_squared_error',cv=5,verbose = 2,random_state = 42,n_jobs = 1) 
hyp.fit(x_train,y_train)

y_pred = hyp.predict(x_test)
print(y_pred)

sns.histplot(y_test-y_pred)


file = open('file.pkl','wb')
pickle.dump(hyp,file)
