from audioop import minmax
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler




class dataset:

    data = None
    X_train = None
    Y_train = None

    def __init__(self, path) -> None:
        self.data = pd.read_csv(path)
        self.X_train = self.data.iloc[:, 0:-1]
        self.Y_train = self.data.iloc[:, -1]

    def getdataSet(self):
        return self.data
    def get_Train(self):
        return self.X_train, self.Y_train

    def getFeaturs(self):
        return self.data.columns.values

    def getTypes(self):
        return self.data.dtypes
    
    def encode(self):
        le = LabelEncoder()

        types = dict(self.X_train.dtypes)
        
        print(type(types))
        notNumbers = [x for x,y in types.items() if y != 'float64']
        print(notNumbers)
        
        for col in notNumbers:
            self.X_train[col] = le.fit_transform(self.X_train[col])


        for col in self.X_train.columns:
            mean = int(self.X_train[col].mean(skipna=True))
            self.X_train[col] = self.X_train[col].replace(np.NaN, mean)

    def scale(self):
        
        scale = MinMaxScaler()
        self.X_train = pd.DataFrame(scale.fit_transform(self.X_train))
        self.X_train.columns = self.getFeaturs()[:-1]

ds = dataset("files/train.csv")

ds.encode()

ds.scale()

X_train, Y_train = ds.get_Train()

print(X_train)