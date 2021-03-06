# -*- coding: utf-8 -*-
"""
Created on Sat May 25 11:37:45 2019

@author: DELL
"""
__username__ = "aishitdua"
import numpy as np
import tensorflow as tf
from tensorflow import keras
model = keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs = np.array([-1.0,0.0,1.0,2.0,3.0,4.0],dtype = float)
ys = np.array([-3.0,-1.0,1.0,3.0,5.0,7.0],dtype = float)
model.fit(xs,ys,epochs=500) #running for 500 epochs since it is a small dataset 
print(model.predict([70.0]))