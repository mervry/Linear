# -*- coding: utf-8 -*-
"""
Created on Mon Oct 25 17:20:49 2021

"""
import pandas as pd

veriler=pd.read_csv("MonthAndSales.csv")

X=veriler.iloc[:,0:1]
Y=veriler.iloc[:,1:2]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.33,random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)
tahmin=lr.predict(x_test)
print(tahmin)

print("görselleştirme")

import matplotlib.pyplot as plt
import numpy as np

x_train=x_train.sort_index()
y_train=y_train.sort_index()

'''
plt.plot(x_train, y_train)
'''

plt.scatter(x_train, y_train, color="blue")
plt.plot(x_test,tahmin, color="red")

plt.title("Aylara göre satış")
plt.xlabel("Ay")
plt.ylabel("Satış")
plt.show()



print("-------poly------")

from sklearn.preprocessing import PolynomialFeatures

poly=PolynomialFeatures(degree=4)
x_train_poly=poly.fit_transform(x_train)
print(x_train_poly)
x_test_poly=poly.transform(x_test)

from sklearn.linear_model import LinearRegression
lr2=LinearRegression()
lr2.fit(x_train_poly, y_train)
tahmin2=lr2.predict(x_test_poly)
print(tahmin2)

print ("poly görselleştirme")

plt.scatter(X,Y, color="purple")
plt.plot(X, lr2.predict(poly.fit_transform(X)), color="blue")
plt.title("Aylara göre satış")
plt.xlabel("Ay")
plt.ylabel("Satış")
plt.show
