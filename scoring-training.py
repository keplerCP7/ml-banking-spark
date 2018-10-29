
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle


# In[3]:


from keras.models import Model
from keras.models import Sequential
from keras.layers import Dense, Activation, Reshape, merge, Embedding, Input, Dropout, BatchNormalization
from keras.layers import concatenate
from keras import layers
from keras import optimizers
from keras import backend as K
import tensorflow as tf
K.clear_session()
tf.reset_default_graph()


# In[4]:


x_train_df=pd.read_csv("./x_train.csv")
y_train_df=pd.read_csv("./y_train.csv")

x_val_df=pd.read_csv("./x_val.csv")
y_val_df=pd.read_csv("./y_val.csv")


# In[5]:


num_features=x_train_df.shape[1] 


# In[6]:


DropoutRate=0.5
dim=len(x_train_df)


model = Sequential()
model.add(Dense(600, input_dim=num_features, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(DropoutRate))

model.add(Dense(300, input_dim=num_features, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(DropoutRate))


model.add(Dense(150, input_dim=num_features, kernel_initializer='uniform'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(DropoutRate))

model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='mse', optimizer='Adam', metrics=['accuracy'])


# In[7]:


# calculate the batch size
num_steps=200
training_epochs=4

x_train=x_train_df.as_matrix()
y_train=y_train_df.as_matrix()
x_val=x_val_df.as_matrix()
y_val=y_val_df.as_matrix()


batch_size= x_train_df.shape[0]// num_steps

print("bacht size", batch_size)

h=model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=training_epochs, batch_size=batch_size)

model.save( './model.h5')

