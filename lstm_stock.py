# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 18:04:55 2020

@author: Ted
"""
# stock price prediction using LSTM with multi inputs
#%%

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTMCell, RNN
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#%% define functions
def make_dataset(raw_data, n_data=100, n_prev=25):
    """
    make dataset to predict a value from past 'n_prev' data
    """
    trainlist = []
    testlist = []
    for i in range(n_data-n_prev):
        trainlist.append(raw_data[i:i+n_prev])
        testlist.append(raw_data[i+n_prev])
    return np.array(trainlist, dtype='float32'), np.array(testlist, dtype='float32')

def scaler(data):
    """
    scaling data
    """
    sscaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    sscaler.fit(data)
    return sscaler.transform(data)

def lstm_model(n_hidden):
    """
    LSTM model
    """
    model=Sequential()
    model.add(RNN(LSTMCell(n_hidden)))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('linear'))
    optim = Adam(lr=0.001)
    model.compile(loss='mean_squared_error', optimizer=optim)
    return model

#%% import data
df = pandas_datareader.data.DataReader("NVDA","yahoo","2018-01-01","2020-01-01")
n_data = len(df)
data = np.array(df.loc[:, ['Adj Close','Open','High','Low']].head(n_data)).squeeze()

#%% scaling and make dataset to input into LSTM
if len(data.shape)>1:
    # multiple dimension inputs
    data = scaler(data)
    x, y = make_dataset(data, n_data)
else:
    #single dimension input
    data = (data-np.min(data)) / (np.max(data)-np.min(data))
    x, y = make_dataset(data, n_data)    
    x = x[...,np.newaxis]
    y = y[...,np.newaxis]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

#%% model training
n_hidden = 100
model = lstm_model(n_hidden)
early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=10)
history = model.fit(X_train, Y_train, batch_size=64, epochs=50, validation_split=0.1, callbacks=[early_stopping])
#%%
plt.figure()
plt.plot(history.history['loss'])
#%%
pred = model.predict(X_test)
plt.figure()
plt.plot(pred,label='predicted')
plt.plot(Y_test[:,0],label='correct')
plt.legend()
