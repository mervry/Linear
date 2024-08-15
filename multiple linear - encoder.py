# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 18:53:35 2021

@author: URAY
"""

import pandas as pd        #veri okumak için
import matplotlib.pyplot as plt
import numpy as np         #sayısal veri okuması için

veriler=pd.read_csv("insurance.csv")

#kategorik-nümerik dönüşümü için (encoder)

sex=veriler.iloc[:,1:2]
smoker=veriler.iloc[:,4:5]
region=veriler.iloc[:,5:6]

#1.yol
cat_variables=veriler[["sex", "smoker", "region"]]  #catli olanı kendimiz koyuyoruz
cat_dummies=pd.get_dummies(cat_variables, drop_first=True) #yazıyı nümeriğe dönüştürmek için

#veri dosyasını tekrar oluştur
#iki data frame i birleşirmek için pd.concat

df1=veriler.iloc[:,0:1]      #df=data frame
df2=pd.concat([df1, cat_dummies.iloc[:,0:1]], axis=1)
df3=pd.concat([df2, veriler.iloc[:,2:3]], axis=1)
df4=pd.concat([df3,veriler.iloc[:,3:4]], axis=1)
df5=pd.concat([df4,cat_dummies.iloc[:,1:5]], axis=1)
veriler2=pd.concat([df5,veriler.iloc[:,6:7]], axis=1)

'''
preprocessing=ön işleme(datayı hazır hale getirmek)
cat_variables=veriler[["sex", "smoker","region"]]
cat_dummies=pd.get_dummies(cat_variables,dropfirst=True)
cat_dummies.head()

'''
X=veriler2.iloc[:,0:8] #Bağımsız değişkenler
Y=veriler2.iloc[:,8:9] #Bağımlı değişken

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.33, random_state=0) #%33 istersen size ı farklı da seçebilirsin/random state overfitting i yani ezberi önlüyor ve her başladığında aynı yerden başlamısını sağlıyor 

#Multi Linear Regression
from sklearn.linear_model import LinearRegression #eğitmek için kullandığımız algoritma 
lr=LinearRegression() #kısaltma için
lr.fit(x_train, y_train)#eğittik

tahmin=lr.predict(x_test)
print(tahmin)

print("evaluation") #değerlendirme yaptık 
from sklearn.metrics import r2_score
print(r2_score(y_test, tahmin)) 

