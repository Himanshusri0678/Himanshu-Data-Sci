#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("comcast_data.csv")


# In[5]:


df.head()


# In[6]:


print(df.isnull().sum())


# In[7]:


df.describe(include='all')


# In[8]:


df.shape


# In[9]:


df= df.drop(['Ticket #','Time'], axis=1)


# In[10]:


df.head()


# In[14]:


df['Date_month_year'] = df['Date_month_year'].apply(pd.to_datetime)
df = df.set_index('Date_month_year')


# In[15]:


months = df.groupby(pd.Grouper(freq="M")).size().plot()
plt.xlabel("MONTHS")
plt.ylabel("FREQUENCY")
plt.title("MONTHLY TREND CHART")


# In[16]:


df['Date'].value_counts(dropna=False)[:8]


# In[17]:


df=df.sort_values(by='Date')
plt.figure(figsize=(6,6))
df['Date'].value_counts().plot()
plt.xlabel("Date")
plt.ylabel("FREQUENCY")
plt.title("DAILY TREND CHART")


# In[18]:


df['Customer Complaint'].value_counts(dropna=False)[:9].plot.bar()


# In[20]:


internet_issues1=df[df['Customer Complaint'].str.contains("network")].count()


# In[21]:


internet_issues2=df[df['Customer Complaint'].str.contains("speed")].count()


# In[22]:


internet_issues3=df[df['Customer Complaint'].str.contains("data")].count()


# In[25]:


internet_issues4=df[df['Customer Complaint'].str.contains("internet")].count()


# In[27]:


billing_issues1=df[df['Customer Complaint'].str.contains("bill")].count()


# In[28]:


billing_issues2=df[df['Customer Complaint'].str.contains("billing")].count()


# In[29]:


billing_issues3=df[df['Customer Complaint'].str.contains("charges")].count()


# In[30]:


service_issues1=df[df['Customer Complaint'].str.contains("service")].count()


# In[31]:


service_issues2=df[df['Customer Complaint'].str.contains("customer")].count()


# In[9]:


total_internet_issues=internet_issues1+internet_issues2+internet_issues3+internet_issues4
print(total_internet_issues)


# In[36]:


total_billing_issues=billing_issues1+billing_issues2+billing_issues3
print(total_billing_issues)


# In[35]:


total_service_issues=service_issues1+service_issues2
print(total_service_issues)


# In[37]:


other_issues=2224-(total_internet_issues+total_billing_issues+total_service_issues)
print(other_issues)


# In[4]:


df.Status.unique()


# In[5]:


df["newStatus"] = ["Open" if Status=="Open" or Status=="Pending" else "Closed" for Status in df["Status"]]
df = df.drop(['Status'], axis=1)
df


# In[6]:


#5.which state has the maximum complaints


# In[7]:


df.groupby(["State"]).size().sort_values(ascending=False)[:5]


# In[13]:


#Insights - Georgia has maximun complaints


# In[9]:


#6.State wise bar chart


# In[11]:


Status_complaints = df.groupby(["State","newStatus"]).size().unstack()
print(Status_complaints)


# In[12]:


Status_complaints.plot.bar(figsize=(10,10), stacked=True)


# In[14]:


#insights georgia has the highest numbers of complaints



# In[15]:


#7.State with highest numbers of unresolved complaints



# In[16]:


print(df['newStatus'].value_counts())


# In[19]:


unresolved_data = df.groupby(["State",'newStatus']).size().unstack().fillna(0).sort_values(by='Open' ,ascending=False)
unresolved_data['Unresolved_cmp_prct'] = unresolved_data['Open']/unresolved_data['Open'].sum()*100
print(unresolved_data)


# In[20]:


unresolved_data.plot()


# In[21]:


#insights Georgia has the maximum numbers of uresolved complaints.



# In[22]:


#8. % of resolved cases (internet,customercare calls)



# In[25]:


resolved_data = df.groupby(['Received Via','newStatus']).size().unstack().fillna(0)
resolved_data['resolved'] = resolved_data['Closed']/resolved_data['Closed'].sum()*100
resolved_data['resolved']


# In[26]:


resolved_data.plot(kind="bar", figsize=(8,8))


# In[ ]:


#insights 50.61% complaints resolved for customer care call and 49.39% for received via internet.


