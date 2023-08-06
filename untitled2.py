import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv(r'C:\machine learning\archive (1)\\housing.csv')
dataset.info()
dataset.dropna(inplace = True)

x  = dataset.drop(['median_house_value'],axis = 1)
y = dataset['median_house_value']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)

train_data = x_train.join(y_train)
train_data.hist(figsize =  (15, 8))


train_data.corr(numeric_only = True)
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot = True,cmap="YlGnBu")
print(train_data.ocean_proximity.value_counts()) 

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis = 1)
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot = True,cmap="YlGnBu")

#feature engineeering
train_data['bedroom_ratio'] = train_data['total_bedrooms'] / train_data['total_rooms']
train_data['household_rooms'] = train_data['total_bedrooms'] / train_data['total_rooms']
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(), annot = True,cmap="YlGnBu")

##fiting median house value in x_train and y_train
x_train,y_train = train_data.drop(['median_house_value'],axis = 1),train_data['median_house_value']

test_data = x_test.join(y_test)
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'],axis = 1)
test_data['bedroom_ratio'] = test_data['total_bedrooms'] / test_data['total_rooms']
test_data['household_rooms'] = test_data['total_bedrooms'] / test_data['total_rooms']

##fiting median house value in x_test and y_test
x_test,y_test = test_data.drop(['median_house_value'],axis = 1),test_data['median_house_value']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators= 1000,random_state=0)
forest.fit(x_train_scaled,y_train)
print(forest.score(x_test_scaled,y_test))



