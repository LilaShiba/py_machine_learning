
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('USA_Housing.csv')


# In[5]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe()


# In[9]:


df.columns


# In[10]:


sns.pairplot(df)


# In[11]:


sns.distplot(df['Price'])


# In[13]:


sns.heatmap(df.corr(), annot=True)


# In[14]:


df.columns


# In[15]:


X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
       'Avg. Area Number of Bedrooms', 'Area Population']]


# In[29]:


y = df['Price'] #what we are trying to predict


# In[27]:


from sklearn.model_selection import train_test_split


# In[30]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)


# In[31]:


from sklearn.linear_model import LinearRegression


# In[32]:


lm = LinearRegression()


# In[34]:


lm.fit(X_train,y_train)


# In[35]:


print(lm.intercept_)


# In[36]:


print(lm.coef_)


# In[37]:


pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])


# In[47]:


predictions = lm.predict(X_test)


# In[48]:


print(predictions)


# In[50]:


plt.scatter(y_test,predictions)


# In[52]:


sns.distplot((y_test-predictions))


# In[53]:


from sklearn import metrics


# In[54]:


metrics.mean_absolute_error(y_test,predictions)


# In[56]:


mse = metrics.mean_squared_error(y_test,predictions)


# In[57]:


np.sqrt(mse)

