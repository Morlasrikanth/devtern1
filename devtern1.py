#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


df = pd.read_csv("HousePricePrediction.xlsx - Sheet1.csv")
print(df.head())


# In[3]:


plt.figure(figsize=(10,6))
sns.histplot(df['SalePrice'],bins=30,kde=True)
plt.title('Distribution of SalePrice')
plt.xlabel('SalePrice')
plt.show()


# In[4]:


numeric_df = df.select_dtypes(include=[np.number])

correlation_matrix = numeric_df.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()


# In[5]:


top_corr_features = correlation_matrix['SalePrice'].sort_values(ascending=False).index[1:4]
print("Top correlated features",top_corr_features)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

features = df[top_corr_features]
target = df['SalePrice']


# In[6]:


print("Missing values in the dataset:\n", df.isnull().sum())


# In[7]:


df = df.dropna()

features = df[top_corr_features]
target = df['SalePrice']

X_train,X_test,y_train,y_test = train_test_split(features,target,test_size = 0.2,random_state = 42)
print('X_train',X_train)
print('X_test',X_test)
print('y_train',y_train)
print('y_test',y_test)


# In[8]:


model = LinearRegression()
model.fit(X_train,y_train)


# In[9]:


y_pred = model.predict(X_test)
print('y_pred',y_pred)


# In[10]:


mse = mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)

print(f"Mean Square Error: {mse}")
print(f"R-squared: {r2}")


# In[ ]:




