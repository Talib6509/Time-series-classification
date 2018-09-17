
# coding: utf-8

# In[101]:


#importing libraries
import pandas as pd #for data representation
import numpy as np #for filling Nan values
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split # splitting the data into train test model
from sklearn.metrics import accuracy_score # for meausring the accuracy 


# In[6]:


#reading the train data
data= pd.read_csv("C:/Users/user/Downloads/mle_task/train.csv",parse_dates=["Timestamp"])


# In[10]:


# setting the timestamp as index
data.set_index(data.Timestamp,inplace=True)


# In[98]:


print(data.head())


# In[24]:


#scalar filtering 
data=data.iloc[:,1:]


# In[25]:


print(data.head())


# In[41]:


#splitting the data into depentdent and independent variables
x=data.iloc[:,:5]
y=data.Label


# In[63]:


#filling the NAN values
data.fillna(method="ffill")


# In[70]:


#splitting the data into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=12)


# In[71]:


#building model
model=XGBClassifier()


# In[72]:


model.fit(x_train,y_train)


# In[73]:


#Predicting 
y_predict=model.predict(x_test)


# In[99]:


#checking the accuracy of the model
print(accuracy_score(y_test,y_predict))


# In[79]:


# reading the test data
data1=pd.read_csv("C:/Users/user/Downloads/mle_task/test.csv",parse_dates=["Timestamp"])


# In[81]:


data1.set_index(data1.Timestamp,inplace=True)
data1=data1.iloc[:,1:]


# In[97]:


print(data1.head())


# In[102]:


#predictng for test data
y2_predict=model.predict(data1)


# In[100]:


print(y2_predict)

