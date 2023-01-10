#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[2]:


pwd


# In[3]:


df = pd.read_csv('health care diabetes.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# In[7]:


df.shape


# In[8]:


df.describe()


# In[9]:


df.isnull().any()


# In[10]:


df.isnull().sum()


# In[11]:


df.columns


# In[12]:


print((df[['Glucose']]==0).sum())


# In[13]:


print((df[['BloodPressure']]==0).sum())


# In[14]:


print((df[['Insulin']]==0).sum())


# In[15]:


print((df[['BMI']]==0).sum())


# In[16]:


print((df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]==0).sum())


# In[17]:


print((df[['Glucose']]==0).count())


# In[18]:


print((df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]==0).count())


# In[19]:


df.head()


# In[23]:


df [['Pregnancies','Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
'BMI', ]] = df [['Pregnancies','Glucose', 'BloodPressure','SkinThickness', 'Insulin',
'BMI', ]].replace(0,np.NaN)
df.head()


# In[24]:


df['Pregnancies'].fillna(df['Pregnancies'].mean(), inplace = True)


# In[25]:


print(df['Pregnancies'].isnull().sum())


# In[26]:


df.fillna(df.mean(), inplace=True)
df.head()


# In[27]:


print (df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness','Insulin','BMI', 'DiabetesPedigreeFunction', 'Age']].isnull().sum())


# In[28]:


labels = 'Diabetic', 'Non-diabetec'
df.Outcome.value_counts().plot.pie(labels=labels, autopct='%1.1f%%',shadow=True, startangle=150)


# In[29]:


diabetes_agewise = df[df['Outcome']==1]
diabetes_agewise.groupby('Age')['Outcome'].count()


# In[30]:


diabetes_agewise.groupby('Age')['Outcome'].count().plot.pie(autopct='%1.1f%%',shadow=True, startangle=150, figsize=(35,18))


# In[31]:


diabetes_agewise.groupby('Age')['Outcome'].count().plot(kind= 'barh',figsize=(8,15))


# In[32]:


df.hist(figsize=(15,10))


# In[33]:


sns.countplot(df['Glucose'])


# In[34]:


sns.countplot(df['BloodPressure'])


# In[35]:


sns.countplot(df['SkinThickness'])


# In[36]:


sns.countplot(df['Insulin'])


# In[37]:


sns.countplot(df['BMI'])


# In[38]:


sns.countplot(df['Pregnancies'])


# In[39]:


sns.countplot(df['DiabetesPedigreeFunction'])


# In[40]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.Pregnancies)
plt.title("Boxplot for Preg by Outcome")


# In[41]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.Glucose)
plt.title("Boxplot for Glucose by Outcome")


# In[42]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.BloodPressure)
plt.title("Boxplot for BloodPressure by Outcome")


# In[43]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.SkinThickness)
plt.title("Boxplot for SkinThickness by Outcome")


# In[44]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.Insulin)
plt.title("Boxplot for Insulin by Outcome")


# In[45]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.BMI)
plt.title("Boxplot for BMI by Outcome")


# In[46]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.DiabetesPedigreeFunction)
plt.title("Boxplot for DiabetesPedigreeFunction by Outcome")


# In[47]:


plt.figure(figsize=(20, 6))
plt.subplot(1,3,3)
sns.boxplot(x=df.Outcome,y=df.Age)
plt.title("Boxplot for Age by Outcome")


# In[48]:


sns.lmplot(x='Insulin',y='SkinThickness',data=df,fit_reg=False,hue='Outcome')


# In[49]:


sns.lmplot(x='BMI',y='SkinThickness',data=df,fit_reg=False,hue='Outcome')


# In[50]:


sns.lmplot(x='Insulin',y='Glucose',data=df,fit_reg=False,hue='Outcome')


# In[51]:


sns.lmplot(x='Age',y='Pregnancies',data=df,fit_reg=False,hue='Outcome')


# In[52]:


sns.pairplot(df, vars=["Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin","BMI","DiabetesPedigreeFunction", "Age"],hue="Outcome")
plt.title("Pairplot of Variables by Outcome")


# In[53]:


cor = df.corr()
cor


# In[54]:


sns.heatmap(cor)


# In[55]:


plt.subplots(figsize=(10,12))
sns.heatmap(cor,annot=True,cmap='viridis')


# In[56]:


features = df.iloc[:,[0,1,2,3,4,5,6,7]].values
label = df.iloc[:,8].values


# In[57]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(features,
label,test_size=0.2,random_state =10)


# In[58]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


# In[59]:


print(model.score(X_train,y_train))
print(model.score(X_test,y_test))


# In[60]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(label,model.predict(features))
cm


# In[61]:


from sklearn.metrics import classification_report
print(classification_report(label,model.predict(features)))


# In[62]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
probs = model.predict_proba(features)
probs = probs[:, 1]
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(label, probs)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')


# In[63]:


from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier(max_depth=5)
model3.fit(X_train,y_train)


# In[64]:


model3.score(X_train,y_train)


# In[65]:


model3.score(X_test,y_test)


# In[66]:


from sklearn.ensemble import RandomForestClassifier
model4 = RandomForestClassifier(n_estimators=11)
model4.fit(X_train,y_train)


# In[67]:


model4.score(X_train,y_train)


# In[69]:


model4.score(X_test,y_test)


# In[70]:


from sklearn.svm import SVC
model5 = SVC(kernel='rbf',gamma='auto')
model5.fit(X_train,y_train)


# In[71]:


model5.score(X_test,y_test)


# In[72]:


model5.score(X_test,y_test)


# In[73]:


from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier(n_neighbors=7,metric='minkowski',p = 2)
model2.fit(X_train,y_train)


# In[77]:


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
probs = model2.predict_proba(features)
probs = probs[:, 1]
auc = roc_auc_score(label, probs)
print('AUC: %.3f' % auc)
fpr, tpr, thresholds = roc_curve(label, probs)
print("True Positive Rate - {}, False Positive Rate - {} Thresholds - {}".format(tpr,fpr,thresholds))
plt.plot([0, 1], [0, 1], linestyle='--')
plt.plot(fpr, tpr, marker='.')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")


# In[79]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
probs = model.predict_proba(features)
probs = probs[:, 1]
yhat = model.predict(features)
precision, recall, thresholds = precision_recall_curve(label, probs)
f1 = f1_score(label, yhat)
auc = auc(recall, precision)
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')


# In[80]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
probs = model2.predict_proba(features)
probs = probs[:, 1]
yhat = model2.predict(features)
precision, recall, thresholds = precision_recall_curve(label, probs)
f1 = f1_score(label, yhat)
auc = auc(recall, precision)
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')


# In[81]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
probs = model3.predict_proba(features)
probs = probs[:, 1]
yhat = model3.predict(features)
precision, recall, thresholds = precision_recall_curve(label, probs)
f1 = f1_score(label, yhat)
auc = auc(recall, precision)
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')


# In[82]:


from sklearn.metrics import precision_recall_curve
from sklearn.metrics import f1_score
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
probs = model4.predict_proba(features)
probs = probs[:, 1]
yhat = model4.predict(features)
precision, recall, thresholds = precision_recall_curve(label, probs)
f1 = f1_score(label, yhat)
auc = auc(recall, precision)
ap = average_precision_score(label, probs)
print('f1=%.3f auc=%.3f ap=%.3f' % (f1, auc, ap))
plt.plot([0, 1], [0.5, 0.5], linestyle='--')
plt.plot(recall, precision, marker='.')


# In[ ]:




