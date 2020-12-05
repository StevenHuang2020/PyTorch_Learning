import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

def splitData(X,y, random=False):
    if random:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)#every time same
     
    x_train = torch.Tensor(x_train)
    x_test = torch.Tensor(x_test)
    y_train = torch.Tensor(y_train)
    y_test = torch.Tensor(y_test)  

    print('x_train.shape = ', x_train.shape)
    print('y_train.shape = ', y_train.shape)
    print('x_test.shape = ', x_test.shape)
    print('y_test.shape = ', y_test.shape)
    print('x_train[:5]=', x_train[:5])
    print('x_test[:5]=', x_test[:5])
     
    return x_train, x_test, y_train, y_test

def getCsvDataset(file,skipLines=3, header=None):
    df = pd.read_csv(file, header=header,skiprows=skipLines)
    #print(df.describe().transpose())
    #print(df.head())

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    return splitData(X,y)

def getExcelDataset(file, header=0):
    df = pd.read_excel(file, header=header)
    #print(df.describe().transpose())
    #print(df.head())

    X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
    return splitData(X,y)

def main():
    file = r'./db/fucDatasetReg_1F.csv'
    getCsvDataset(file)

if __name__ == "__main__":
    main()
