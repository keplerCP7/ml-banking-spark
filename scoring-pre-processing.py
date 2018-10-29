
# coding: utf-8

# In[5]:


import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import pickle


train_percent=0.9
df_data = pd.read_csv('s3://vl2-dlk/data-banking-demo/cs-training.csv')
df_data=df_data.iloc[:,1:]


# In[2]:


df_data.fillna(0, inplace=True)


# In[3]:


############# Split training , validation ###########################
long = len(df_data)

seed=[1,long]

np.random.seed(seed)
perm = np.random.permutation(df_data.index)
m = len(df_data.index)
train_end = int(train_percent * m)

train = df_data.iloc[perm[:train_end]]
validate = df_data.iloc[perm[train_end:]]


# In[4]:


x_train=train.iloc[:,1:];
y_train=train.iloc[:,:1];
x_val=validate.iloc[:,1:];
y_val=validate.iloc[:,:1];


# In[5]:


###### Standarization ###############################
print("Start StandardScaler")

e=StandardScaler()

x_train= e.fit_transform(x_train)
x_val= e.transform(x_val)


filehandler = open("/stdscaler.obj","wb")
pickle.dump(e,filehandler)
filehandler.close()

print("End StandardScaler")
#####################################################################


# In[11]:

x_train_df=pd.DataFrame(x_train)
x_train_df.to_csv("./x_train_df.csv")

x_val_df=pd.DataFrame(x_val)
x_val_df.to_csv("./x_val_df.csv")

y_train.to_csv("./y_train.csv")
y_val.to_csv("./y_val.csv")

