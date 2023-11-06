#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt 
import tensorflow as tf


# In[2]:


# here the file name churn_modeling is being replaced by Bank.csv
df=pd.read_csv("Bank.csv")
df.head()


# In[3]:


x=df.iloc[:,3:-1].values
y=df.iloc[:,-1].values 


# In[4]:


print(x)


# In[5]:


print(y)


# #taking care of mising data
# import numpy as np
# from sklearn.impute import SimpleImputer
# imputer =SimpleImputer(missing_values=np.nan,strategy='mean')
# imputer.fit(x[:,1:3])
# x[:,1:3]=imputer.transform(x[:,1:3])

# '''encoding the categorical variable using label encoder
# A label encoder is a commonly used tool in machine learning and data preprocessing.
# Its primary function is to transform categorical data, which consists of labels or
# text values, into numerical values. This is important because many machine learning
# algorithms and models require numerical input data, and label encoding is one way to
# convert categorical data into a suitable format for these algorithms.
# '''

# In[6]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x[:,2]=le.fit_transform(x[:,2])
print(x)
# here we have converted the categorical data into numeric one. MALE=1,FEMALE=0


# as now we have to convert the geography colloum in to numeric one so has there is no relation between france
# germany ,spain or any other country we use the the other method called ONE HOT ENCODING.

# In[8]:


import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x=np.array(ct.fit_transform(x))


# In[9]:


print(x)


# In[10]:


from sklearn.model_selection import train_test_split


# In[11]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[12]:


print(x_train)


# In[13]:


print(x_test)


# FEATURE SCALLING- IT IS THE MOST IMPORTANT STEP WHILE APPLYING THE ARTIFICIAL NEURAL NETWORK WE NEED
# TO STANDARDIZE THE VALUES USING THE INBUILT LIBRARY IN PYTHON.

# In[14]:


#FEATURE SCALING 
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform(x_test)


# In[15]:


print(x_train)


# In[16]:


print(x_test)


# In[ ]:


#BUILDING THE ANN(ARTIFICIAL NEURAL NETWORK) USING TENSORFLOW KERAS MODEL


# In[17]:


ann=tf.keras.models.Sequential()


# In[18]:


ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


# In[19]:


ann.add(tf.keras.layers.Dense(units=6,activation='relu'))


# In[20]:


ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))


# In[21]:


ann.compile(optimizer='adam' ,loss='binary_crossentropy' ,metrics=['accuracy'] ,)


# In[22]:


ann.fit(x_train,y_train,batch_size=32,epochs=100)


# In[26]:


ann.predict([[1,0,0,600,1,40,3,]])


# Important note 1: Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.
# 
# Important note 2: Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.

# In[34]:


y_pred=ann.predict(x_test)
y_pred=(y_pred>0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


# In[36]:


from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
accuracy_score(y_test,y_pred)

'''
[[True Negative (TN)  False Positive (FP)]
 [False Negative (FN)  True Positive (TP)]]
True Positive (TP): The number of correct positive predictions.
True Negative (TN): The number of correct negative predictions.
False Positive (FP): The number of incorrect positive predictions (Type I error).
False Negative (FN): The number of incorrect negative predictions (Type II error).
'''


# In[ ]:




