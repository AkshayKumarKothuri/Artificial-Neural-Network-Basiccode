#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


dataset = pd.read_csv('Churn_Modelling.csv')


# In[3]:


dataset


# In[4]:


dataset.isnull().any()


# In[5]:


x = dataset.iloc[:,3:-1].values
y = dataset.iloc[:,13].values


# In[6]:


x


# In[7]:


x[:,1]


# In[8]:


from sklearn.preprocessing import LabelEncoder
lb1 = LabelEncoder()
x[:,1] = lb1.fit_transform(x[:,1])


# In[9]:


lb2 = LabelEncoder()
x[:,2] = lb1.fit_transform(x[:,2])


# In[10]:


from sklearn.preprocessing import OneHotEncoder
o1 = OneHotEncoder(categorical_features = [1])
x = o1.fit_transform(x).toarray()


# In[11]:


x=x[:,1:]


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2,random_state = 0)


# In[13]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[14]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[15]:


model=Sequential()


# In[16]:


x_train.shape


# In[17]:


#input layer
model.add(Dense(input_dim=11,init="random_uniform",activation='relu',output_dim=6))


# In[18]:


#hidden layer
model.add(Dense(init="random_uniform",activation='relu',output_dim=6))


# In[19]:


#output layer
model.add(Dense(output_dim=1,init='random_uniform',activation='sigmoid'))


# In[20]:


#training the model in ANN
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[21]:


#Fitting model values to train them.
model.fit(x_train,y_train,epochs=50,batch_size=32)


# In[31]:


y_pred=model.predict(x_test)


# In[32]:


y_pred


# In[37]:


y_pred.max()


# In[38]:


y_pred.min()


# In[66]:


y_p=model.predict((np.array([[756,0,1,1,24,8,11500,2,1,1,10000]])))


# In[67]:


y_p


# In[68]:


(y_p > 0.5)

