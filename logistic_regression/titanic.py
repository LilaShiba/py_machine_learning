
# coding: utf-8

# In[109]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[110]:


train = pd.read_csv('titanic_train.csv')


# In[111]:


train.head()


# In[112]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[113]:


sns.set_style('whitegrid')


# In[114]:


sns.countplot(x='Survived', data=train)


# In[115]:


sns.countplot(x='Survived', data=train, hue='Sex')


# In[116]:


sns.countplot(x='Survived', data=train, hue='Pclass')


# In[117]:


sns.distplot(train['Age'].dropna(),kde=False, bins=30)


# In[118]:


train_corr = train.corr()


# In[119]:


sns.heatmap(train_corr, annot=True)


# In[120]:


plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)


# In[121]:


def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[122]:


train['Age'] = train[['Age', 'Pclass']].apply(impute_age,axis=1)


# In[123]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[124]:


train.drop('Cabin', axis=1, inplace=True)


# In[125]:


train.head()


# In[126]:


train.dropna(inplace=True)


# In[127]:


sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


# In[128]:


sex = pd.get_dummies(train['Sex'],drop_first=True)


# In[129]:


embark = pd.get_dummies(train['Embarked'], drop_first=True)


# In[130]:


train = pd.concat([train,sex, embark], axis=1)


# In[131]:


train_corr = train.corr()


# In[132]:


plt.figure(figsize=(10,7))
sns.heatmap(train_corr, annot=True)


# In[133]:


train.head()


# In[134]:


train.drop(['Sex', 'Name', 'Embarked','Ticket','PassengerId'], axis=1, inplace=True)


# In[135]:


train.head()


# In[136]:


train.drop(['PassengerId', 'Cabin'], axis=1, inplace=True)


# In[137]:


titanic = train


# In[138]:


train_corr= train.corr()
plt.figure(figsize=(8,10))
sns.heatmap(train_corr, annot = True)


# In[139]:


X = train.drop('Survived', axis=1)
y = train['Survived']


# In[140]:


from sklearn.cross_validation import train_test_split


# In[141]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)


# In[142]:


from sklearn.linear_model import LogisticRegression


# In[143]:


logmodel = LogisticRegression()

train.head()
# In[144]:


logmodel.fit(X_train,y_train)


# In[146]:


predictions = logmodel.predict(X_test)


# In[147]:


from sklearn.metrics import classification_report


# In[148]:


print(classification_report(y_test, predictions))


# In[149]:


from sklearn.metrics import confusion_matrix


# In[151]:


confusion_matrix(y_test,predictions)

