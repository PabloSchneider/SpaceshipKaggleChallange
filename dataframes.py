from audioop import minmax
from random import random
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

from sklearn.ensemble import ExtraTreesClassifier

from sklearn.model_selection import train_test_split


import heapq


class dataset:

    X = None
    Y_train = None
    features = []
    pred_df = []
    erg_Label = ''

    def __init__(self, path) -> None:
        data = pd.read_csv(path)
        self.X = data.iloc[:, 0:-1]
        self.Y_train = data.iloc[:, -1]

    
    def get_Train(self):
        return self.X, self.Y_train

    def getFeaturs(self):
        return self.X.columns.values

    def getTypes(self):
        return self.X.dtypes
    
    #encodes non numeric datatypes to Floats
    def encode(self):
        le = LabelEncoder()

        types = dict(self.X.dtypes)
        
        notNumbers = [x for x,y in types.items() if y != 'float64']
        
        for col in notNumbers:
            self.X[col] = le.fit_transform(self.X[col])


        for col in self.X.columns:
            mean = int(self.X[col].mean(skipna=True))
            self.X[col] = self.X[col].replace(np.NaN, mean)

    #fits the every number between -1 and 1
    def scale(self):
        
        scale = MinMaxScaler()
        self.X = pd.DataFrame(scale.fit_transform(self.X))
        self.X.columns = self.getFeaturs()


    #Returns the n most imported features
    def feature_Extraction(self, n, estomators):

        model = ExtraTreesClassifier(n_estimators=estomators)

        model.fit(self.X, self.Y_train)

        features = self.getFeaturs()
        erg = model.feature_importances_


        res = {features[i]:erg[i] for i in range(len(features))}
        print(res)
        
        self.features = heapq.nlargest(n, res, key=res.get)

    def drop_Features(self):

        drop = [ x for x in self.getFeaturs() if x not in self.features]
        self.X = self.X.drop(drop, axis=1)

    def get_training_test(self):
        x_train,x_test, y_train, y_test = train_test_split(self.X, self.Y_train, test_size=0.2, random_state=42)
        return x_train, x_test, y_train, y_test

    def run_default(self,n ,estomators):
        self.encode()
        self.scale()
        self.feature_Extraction(n, estomators)
        self.drop_Features()

    def init_test_data(self, path, sub_Label):

        testset = pd.read_csv(path)
        self.X = testset
        self.pred_df = pd.DataFrame(testset[sub_Label])
        print(self.pred_df)

    def run_test_deafault(self):
        self.encode()
        self.scale()
        self.drop_Features()
        return self.X
    
    def to_csv(self, pred, pred_Label, path):
        self.pred_df[pred_Label] = pred
        print(self.pred_df)
        self.pred_df.to_csv(path, index=False)
    