#!/usr/bin/env/ python3

from pyexpat import features
from tokenize import Number
from unicodedata import numeric
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


#preprozessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing  import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#feature selection
from sklearn.ensemble import ExtraTreesClassifier

#ml
from sklearn.compose import ColumnTransformer
from sklearn.neighbors import KNeighborsClassifier

#testing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score




dataset = pd.read_csv("files/train.csv")




colnames = ['PassengerId','Cabin','HomePlanet', 'Destination', 'Name']

le = LabelEncoder()
for col in colnames:
    dataset[col] = le.fit_transform(dataset[col])


for col in dataset.columns:
    mean = int(dataset[col].mean(skipna=True))
    dataset[col] = dataset[col].replace(np.NaN, mean)
    print('meh')

#prep data
X_train = dataset.iloc[:, 0:-1]
y_train = dataset.iloc[:, -1]

X_train = X_train.drop(['PassengerId','HomePlanet', 'Destination', 'VIP', 'RoomService', 'FoodCourt','ShoppingMall','Spa','Name'], axis=1)

#scaling features





scaler = MinMaxScaler()


X_train = scaler.fit_transform(X_train)
#X_train[['Age', 'RoomService', 'ShoppingMall','FoodCourt', 'Spa', 'VRDeck']] = scaler.fit_transform(X_train[['Age', 'RoomService', 'ShoppingMall','FoodCourt', 'Spa', 'VRDeck']])
#X_test[['Age', 'RoomService', 'ShoppingMall','FoodCourt', 'Spa', 'VRDeck']] = scaler.fit_transform(X_test[['Age', 'RoomService', 'ShoppingMall','FoodCourt', 'Spa', 'VRDeck']])


'''
#feature extraction

model = ExtraTreesClassifier(n_estimators=10)

model.fit(X_train, y_train)


#Cryo, Cabin, Age
print(model.feature_importances_)
'''
#Define teh model

classifier = KNeighborsClassifier(n_neighbors=83, p= 2, metric='euclidean')

classifier.fit(X_train, y_train)

#predicting the new test.csv


datasetTest = pd.read_csv("files/test.csv")



datasetTest = datasetTest.drop(['HomePlanet', 'Destination', 'VIP', 'RoomService', 'FoodCourt','ShoppingMall','Spa','Name'], axis=1)

colnames = ['Cabin']

le = LabelEncoder()
for col in colnames:
    datasetTest[col] = le.fit_transform(datasetTest[col])

for col in datasetTest.columns[1:]:
    mean = int(datasetTest[col].mean(skipna=True))
    datasetTest[col] = datasetTest[col].replace(np.NaN, mean)
    print('meh')

datasetTest[['CryoSleep', 'Cabin', 'VRDeck' ,'Age']] = scaler.fit_transform(datasetTest[['CryoSleep', 'Cabin', 'VRDeck' ,'Age']])

print(datasetTest)

dataforNN = datasetTest.copy()
dataforNN = dataforNN.drop(['PassengerId'], axis=1)


print(dataforNN)

pred = classifier.predict(dataforNN)

datasetTest['Transported'] = pred


datasetTest = datasetTest.drop(['CryoSleep','Cabin', 'Age', 'VRDeck'], axis=1)

print(datasetTest)

datasetTest.to_csv('files/submissions.csv', index=False)


