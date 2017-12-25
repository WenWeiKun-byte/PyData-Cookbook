from __future__ import print_function

import numpy as np
import pandas as pd
import time
import datetime
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, Imputer, OneHotEncoder
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

from hyperopt import Trials, STATUS_OK, tpe
from keras.datasets import mnist
from keras.layers.core import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback

from hyperas import optim
from hyperas.distributions import choice, uniform, conditional

np.random.seed(20171212)


def data():
    path = '../datasource/APP埋点信用/%s'
    X = np.load(path%'x_before.npy')
    Y = np.load(path%'y_before.npy')
    print (X.shape)
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=20171214)
    return x_train, y_train, x_test, y_test
	
	
def create_model(x_train, y_train, x_test, y_test):
    
    model = Sequential()
    model.add(Dense(256, input_shape=(174,)))
    model.add(Activation('relu'))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense({{choice([256,512,1024])}}))
    model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense({{choice([256,512,1024])}}))
    model.add(Dropout({{uniform(0,1)}}))
    
    # if we choose 'four', we add an additional foutth layer
    if conditional({{choice(['three', 'four'])}}) == 'four':
        model.add(Dense(100))
        model.add({{choice([Dropout(0.5), Activation('linear')])}})
        model.add(Activation('relu'))
    
    # 输出层
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    print (model.summary())
    
    model.compile(
        loss='binary_crossentropy', 
        metrics=['accuracy'],
        optimizer='sgd')  # 
    
    model.fit(x_train, y_train, batch_size={{choice([64,128,256,512,1024,2048])}},epochs=10, verbose=2, validation_data=(x_test, y_test))
    score, acc = model.evaluate(x_test, y_test, verbose=0)
    print ('Test acc:', acc)
    return {'loss': score, 'status': STATUS_OK, 'model':model}
	
def run():
    best_run, best_model = optim.minimize(model=create_model,
                                         data=data,
                                         algo=tpe.suggest,
                                         max_evals=20,
                                         trials=Trials(),
                                         notebook_name='模型')
    X_train, Y_train, X_test, Y_test = data()
    print (best_model.evaluate(X_test, Y_test))
    print (best_run)
    return best_run, best_model
	
best_run, best_model = run()