#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
variance = VarianceThreshold(threshold=0)
from sklearn.preprocessing import LabelEncoder
label = LabelEncoder


# In[3]:


train = pd.read_csv("train.csv")


# In[4]:


train.head()


# In[5]:


test = pd.read_csv("test.csv")
test.head()


# In[6]:


test.describe()


# In[7]:


train.isnull().sum()


# In[8]:


train_target = train["y"]
train_data = train.drop(["y","ID"],axis=1)
train_data.head(5)


# In[9]:


train_data.var().sort_values().head(15)


# In[20]:


train_data_without_zero_var = variance.fit_transform(train_data.iloc[:,9:])
train_data_without_zero_var


# In[21]:


labeled_data = train_data.iloc[:,0:8]
labeled_data.head()


# In[22]:


labeled_data.nunique()


# In[23]:


labeled_data1 = labeled_data.apply(label().fit_transform)
labeled_data1.head()


# In[24]:


labeled_data1.var()


# In[25]:


train_data_Zero_var_final = pd.DataFrame(train_data_without_zero_var)
train_data_Zero_var_final.head()


# In[26]:


final_train_data = pd.concat([labeled_data1,train_data_Zero_var_final],axis=1)
final_train_data.head()


# In[27]:


final_train_data.isnull().any()


# In[28]:


test = test.drop(['ID'],axis=1)
test.head()


# In[29]:


test.head()


# In[30]:


test.nunique()


# In[31]:


test.isnull().any()


# In[32]:


test.var().sort_values().head(15)


# In[33]:


test_without_zero_var = variance.transform(test.iloc[:,9:])
test_without_zero_var


# In[34]:


test_without_zero_var_final = pd.DataFrame(test_without_zero_var)
test_without_zero_var_final.head()


# In[35]:


labeled_data1 = test.iloc[:,0:8]
labeled_data1.head()


# In[36]:


test_label = labeled_data1.apply(label().fit_transform)
test_label.head()


# In[37]:


test_data_final = pd.concat([test_label, test_without_zero_var_final],axis=1)
test_data_final.head(5)


# In[38]:


from sklearn.model_selection import train_test_split


# In[39]:


x_train, x_test, y_train, y_test = train_test_split(final_train_data, train_target, random_state=42, test_size=0.3)


# In[40]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[41]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)


# In[46]:


x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
test_data_final = pca.transform(test_data_final)


# In[43]:


from sklearn import svm
from sklearn.metrics import r2_score, mean_squared_error


# In[44]:


from xgboost import XGBRegressor
xgbr = XGBRegressor(random_state=42)


# In[45]:


model = xgbr.fit(x_train, y_train)


# In[47]:


ypred_test = model.predict(x_test)
ypred_test


# In[48]:


ypred_train = model.predict(x_train)
ypred_train


# In[49]:


print(r2_score(ypred_train, y_train))


# In[50]:


print(mean_squared_error(ypred_train, y_train))


# In[51]:


test_data_final_prediction = model.predict(test_data_final)
test_data_final_prediction


# In[52]:


prediction = pd.DataFrame({'ytest' :y_test, 'ypred':ypred_test})


# In[53]:


plt.plot(prediction['ytest'],color='red')
plt.plot(prediction['ypred'],color='blue')
plt.show()


# In[ ]:




