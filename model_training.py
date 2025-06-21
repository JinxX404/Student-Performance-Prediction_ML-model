import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler


dataset = pd.read_csv('student-mat.csv')

unrelated_cols=['school','sex','address', 'famsize', 'Medu','Fedu','Mjob','Fjob','reason','guardian',
      'traveltime','schoolsup','activities','nursery', 'goout']
dataset.drop(unrelated_cols, inplace=True, axis=1)

dataset['Pstatus']=(dataset['Pstatus']=='T').astype(int)
dataset['famsup']=(dataset['famsup']=='yes').astype(int)
dataset['paid']=(dataset['paid']=='yes').astype(int)
dataset['higher']=(dataset['higher']=='yes').astype(int)
dataset['internet']=(dataset['internet']=='yes').astype(int)
dataset['romantic']=(dataset['romantic']=='yes').astype(int)


x = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state = 21)

model = LinearRegression()
# model.fit(x_train, y_train)

scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model.fit(x_train_scaled, y_train)


train_score = model.score(x_train_scaled, y_train)
print(f'The Accuracy of training: {round(train_score*100, 2)} %')

test_score = model.score(x_test_scaled, y_test)
print(f'The Accuracy of testing: {round(test_score*100, 2)} %')

y_pred=model.predict(x_test_scaled)