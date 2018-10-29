
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

#e=StandardScaler()

#x_train= e.fit_transform(x_train)
#x_val= e.transform(x_val)


#filehandler = open("/stdscaler.obj","wb")
#pickle.dump(e,filehandler)
#filehandler.close()

print("End StandardScaler")
#####################################################################


# In[11]:


import s3fs

data_path= 's3://vl2-dlk/data-banking-demo/'

Access_Key = 'AKIAITG62SZQZTTUYVNQ'
Access_Secret_Key = '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'


x_train_df=pd.DataFrame(x_train)
#x_train_df.to_csv("s3://vl2-dlk/data-banking-demo/x_train_df.csv")

bytes_to_write = x_train_df.to_csv(None,sep=';', index=False, header=False).encode()
fs = s3fs.S3FileSystem(key=Access_Key, secret=Access_Secret_Key)
with fs.open(data_path + 'x_train_df.csv' , 'wb') as f:
    f.write(bytes_to_write)

x_val_df=pd.DataFrame(x_val)
#x_val_df.to_csv("./x_val_df.csv")
with fs.open(data_path + 'x_val_df.csv' , 'wb') as f:
    f.write(bytes_to_write)

with fs.open(data_path + 'y_train.csv' , 'wb') as f:
    f.write(bytes_to_write)
    
with fs.open(data_path + 'y_val.csv' , 'wb') as f:
    f.write(bytes_to_write) 

