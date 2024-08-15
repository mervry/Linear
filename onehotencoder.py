# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 17:38:46 2021

@author: URAY
"""

import pandas as pd

veriler=pd.read_csv("deneme.csv")
outlook=veriler.iloc[:,0:1].values  #labelencoder ve onehotencoder yapabilmek için array şeklinde ihtiyacımız var, sonra dataframe dönüştürcez 

from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

outlook[:,0]=lb.fit_transform(outlook[:,0])  #böyle bırakırsak makine sayısal değerler arasında ilişki olduğunu sanıyor büyüktür küçüktür gibi o yüzden onehotecoder yapmalıyız
#outlook2=lb.fit_transform(outlook[:,0])

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
outlook=ohe.fit_transform(outlook).toarray()

print("----data frame for outlook---")
outlook=pd.DataFrame(data=outlook, index=range(14), columns=["overcast", "rainy", "sunny"])  #sadece sütünü yazıyoruz(14), dataların sıralaması mutlaka aynı olmalı ilk haliyle


windy_play=veriler.iloc[:,3:5]
windy_play2=windy_play.apply(lb.fit_transform)  #labelencoderı kullanarak eğit ve dönüştür
 
df1=pd.concat([outlook, veriler.iloc[:,2:3]], axis=1)
X=pd.concat([df1, windy_play2], axis=1)
Y=veriler.iloc[:,1:2]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=0)  #her zaman önce inputu sonra outputu yaz

print("-----linear regression-----")

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)  #makineyi eğitiyoruz
tahmin1=lr.predict(x_test)
print(tahmin1)

from sklearn.metrics import r2_score
print(r2_score(y_test, tahmin1)*0.1)

print("------SVM-SVR------")
print("linear-svr")
from sklearn.svm import SVR
svr=SVR(kernel="linear")
svr.fit(x_train, y_train)
tahmin2=svr.predict(x_test)
print(tahmin2)

print("poly-svr")
svr2=SVR(kernel="poly",degree=2)
svr2.fit(x_train, y_train)
tahmin3=svr2.predict(x_test)
print(tahmin3)


print("radial basis function-svr")
svr3=SVR(kernel="rbf")
svr3.fit(x_train, y_train)
tahmin4=svr3.predict(x_test)
print(tahmin4)

print(r2_score(y_test, tahmin2))
print(r2_score(y_test, tahmin3))
print(r2_score(y_test, tahmin4))
