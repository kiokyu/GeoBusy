import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.utils import normalize

data = pd.read_csv('D:\Final_app\Final_app\Last_Hope.csv')
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 30)
model = keras.models.load_model('D:\Final_app\Final_app\Models')


def get_class(datta):
    feture_names = ['Расстояние до ближайшего почтового отделения', 'Тип района', 'Тип здания', 'Проходимость']

    d = {'Категория отделения': []}
    df = pd.DataFrame(data=d)

    features = datta[feture_names].values
    x = np.asarray(features).astype(np.float32)
    x = normalize(x, axis=1)

    results = model.predict(x)

    for i in range(0, len(datta.index)):
        if results[i][0] > results[i][1] and results[i][0] > results[i][2] and results[i][0] > results[i][3]:
            df.loc[i] = 0
        if results[i][1] > results[i][0] and results[i][1] > results[i][2] and results[i][1] > results[i][3]:
            df.loc[i] = 1
        if results[i][2] > results[i][1] and results[i][2] > results[i][0] and results[i][2] > results[i][3]:
            df.loc[i] = 2
        if results[i][3] > results[i][1] and results[i][3] > results[i][2] and results[i][3] > results[i][0]:
            df.loc[i] = 3
    datta['Категория отделения'] = df['Категория отделения']

    return datta


data = data.drop('Категория отделения', axis=1)
data = get_class(data)


def Vibor(n, datta):
    for i in range(0, len(datta.index)):
        if datta.iloc[i]['Категория отделения'] == n:
            return i


data1 = pd.read_csv('D:\Final_app\Final_app\len.csv')
data1 = get_class(data1)
data2 = pd.read_csv('D:\Final_app\Final_app\Fruunz.csv')
data2 = get_class(data2)
data3 = pd.read_csv('D:\Final_app\Final_app\pervmai.csv')
data3 = get_class(data3)
data4 = pd.read_csv('D:\Final_app\Final_app\pervorech.csv')
data4 = get_class(data4)
data5 = pd.read_csv('D:\Final_app\Final_app\Sov.csv')
data5 = get_class(data5)

x1 = Vibor(1, data1)
x1 = data1.loc[x1]
x100 = Vibor(1, data2)
x100 = data2.loc[x100]
x101 = Vibor(1, data3)
x101 = data3.loc[x101]
x102 = Vibor(1, data4)
x102 = data4.loc[x102]
x103 = Vibor(1, data5)
x103 = data5.loc[x103]

x2 = x1['Улица почтового отделения']
x3 = x1['Расстояние до ближайшего почтового отделения']
x4 = x1['Тип района']
x5 = x1['Тип здания']
x6 = x1['Проходимость']
x7 = x100['Улица почтового отделения']
x8 = x100['Расстояние до ближайшего почтового отделения']
x9 = x100['Тип района']
x10 = x100['Тип здания']
x11 = x100['Проходимость']
x12 = x101['Улица почтового отделения']
x13 = x101['Расстояние до ближайшего почтового отделения']
x14 = x101['Тип района']
x15 = x101['Тип здания']
x16 = x101['Проходимость']
x17 = x102['Улица почтового отделения']
x18 = x102['Расстояние до ближайшего почтового отделения']
x19 = x102['Тип района']
x20 = x102['Тип здания']
x21 = x102['Проходимость']
x22 = x103['Улица почтового отделения']
x23 = x103['Расстояние до ближайшего почтового отделения']
x24 = x103['Тип района']
x25 = x103['Тип здания']
x26 = x103['Проходимость']

y1 = Vibor(2, data1)
y1 = data1.loc[y1]
y100 = Vibor(2, data2)
y100 = data2.loc[y100]
y101 = Vibor(2, data3)
y101 = data3.loc[y101]
y102 = Vibor(2, data4)
y102 = data4.loc[y102]
y103 = Vibor(2, data5)
y103 = data5.loc[y103]

y2 = y1['Улица почтового отделения']
y3 = y1['Расстояние до ближайшего почтового отделения']
y4 = y1['Тип района']
y5 = y1['Тип здания']
y6 = y1['Проходимость']
y7 = y100['Улица почтового отделения']
y8 = y100['Расстояние до ближайшего почтового отделения']
y9 = y100['Тип района']
y10 = y100['Тип здания']
y11 = y100['Проходимость']
y12 = y101['Улица почтового отделения']
y13 = y101['Расстояние до ближайшего почтового отделения']
y14 = y101['Тип района']
y15 = y101['Тип здания']
y16 = y101['Проходимость']
y17 = y102['Улица почтового отделения']
y18 = y102['Расстояние до ближайшего почтового отделения']
y19 = y102['Тип района']
y20 = y102['Тип здания']
y21 = y102['Проходимость']
y22 = y103['Улица почтового отделения']
y23 = y103['Расстояние до ближайшего почтового отделения']
y24 = y103['Тип района']
y25 = y103['Тип здания']
y26 = y103['Проходимость']

z1 = Vibor(3, data1)
z1 = data1.loc[z1]
z100 = Vibor(3, data2)
z100 = data2.loc[z100]
z101 = Vibor(3, data3)
z101 = data3.loc[z101]
z102 = Vibor(3, data4)
z102 = data4.loc[z102]
z103 = Vibor(3, data5)
z103 = data5.loc[z103]

z2 = z1['Улица почтового отделения']
z3 = z1['Расстояние до ближайшего почтового отделения']
z4 = z1['Тип района']
z5 = z1['Тип здания']
z6 = z1['Проходимость']
z7 = z100['Улица почтового отделения']
z8 = z100['Расстояние до ближайшего почтового отделения']
z9 = z100['Тип района']
z10 = z100['Тип здания']
z11 = z100['Проходимость']
z12 = z101['Улица почтового отделения']
z13 = z101['Расстояние до ближайшего почтового отделения']
z14 = z101['Тип района']
z15 = z101['Тип здания']
z16 = z101['Проходимость']
z17 = z102['Улица почтового отделения']
z18 = z102['Расстояние до ближайшего почтового отделения']
z19 = z102['Тип района']
z20 = z102['Тип здания']
z21 = z102['Проходимость']
z22 = z103['Улица почтового отделения']
z23 = z103['Расстояние до ближайшего почтового отделения']
z24 = z103['Тип района']
z25 = z103['Тип здания']
z26 = z103['Проходимость']


def TipRayon(m):
    if m == 1:
        m = 'спальный'
    if m == 2:
        m = 'деловой'
    if m == 3:
        m = 'пригород'
    return m


def TipBuild(q):
    if q == 1:
        q = 'жилое'
    if q == 2:
        q = 'офисное'
    return q


def Prohodzz(w):
    if w == 1:
        w = 'высокая'
    if w == 2:
        w = 'средняя'
    if w == 2:
        w = 'низкая'
    return w


x4 = TipRayon(x4)
x5 = TipBuild(x5)
x6 = Prohodzz(x6)
x9 = Prohodzz(x9)
x10 = TipRayon(x10)
x11 = TipBuild(x11)
x14 = TipBuild(x14)
x15 = Prohodzz(x15)
x16 = TipRayon(x16)
x19 = TipRayon(x19)
x20 = TipBuild(x20)
x21 = Prohodzz(x21)
x24 = TipBuild(x24)
x25 = Prohodzz(x25)
x26 = TipBuild(x26)

y4 = TipRayon(y4)
y5 = TipBuild(y5)
y6 = Prohodzz(y6)
y9 = Prohodzz(y9)
y10 = TipRayon(y10)
y11 = TipBuild(y11)
y14 = TipBuild(y14)
y15 = Prohodzz(y15)
y16 = TipRayon(y16)
y19 = TipRayon(y19)
y20 = TipBuild(y20)
y21 = Prohodzz(y21)
y24 = TipBuild(y24)
y25 = Prohodzz(y25)
y26 = TipBuild(y26)

z4 = TipRayon(z4)
z5 = TipBuild(z5)
z6 = Prohodzz(z6)
z9 = Prohodzz(z9)
z10 = TipRayon(z10)
z11 = TipBuild(z11)
z14 = TipBuild(z14)
z15 = Prohodzz(z15)
z16 = TipRayon(z16)
z19 = TipRayon(z19)
z20 = TipBuild(z20)
z21 = Prohodzz(z21)
z24 = TipBuild(z24)
z25 = Prohodzz(z25)
z26 = TipBuild(z26)
