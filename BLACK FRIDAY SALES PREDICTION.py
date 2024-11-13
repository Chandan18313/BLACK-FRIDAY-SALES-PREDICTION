#!/usr/bin/env python
# coding: utf-8

# In[1]:


# manipulation data
import pandas as pd
import numpy as np

#visualiation data
import matplotlib.pyplot as plt
import seaborn as sns 
import matplotlib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode, iplot

#default theme
plt.style.use('ggplot')
sns.set(context='notebook', style='darkgrid', palette='colorblind', font='sans-serif', font_scale=1, rc=None)
matplotlib.rcParams['figure.figsize'] =[8,8]
matplotlib.rcParams.update({'font.size': 15})
matplotlib.rcParams['font.family'] = 'sans-serif'


# In[2]:


pip install plotly


# In[3]:


train=pd.read_csv('blacktrain.csv')
test = pd.read_csv('blacktest.csv')
train.head(5)


# In[4]:


train.shape


# In[5]:


train.info()


# In[6]:


train.dtypes.value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.2f%%',shadow=True)
plt.title('TYPES OF DATA');


# In[7]:


# show the numirical values

num_columns = [f for f in train.columns if train.dtypes[f] != 'object']
num_columns.remove('Purchase')
num_columns.remove('User_ID')
num_columns


# In[8]:


# show the categorical values

cat_columns = [f for f in train.columns if train.dtypes[f] == 'object']
cat_columns


# In[9]:


train.describe(include='all')


# In[10]:


#finding missing values 

missing_values=train.isnull().sum()
percent_missing = train.isnull().sum()/train.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame


# In[11]:


missing_values = train.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values.sort_values(inplace=True)
missing_values.plot.pie(explode=[0.1,0.1],autopct='%1.1f%%',shadow=True)
plt.title(' MISSING VALUES IN DATA');


# In[12]:


train.Product_Category_2.value_counts()


# In[13]:


train.Product_Category_2.describe()


# In[14]:


# Replace using median 
median = train['Product_Category_2'].median()
train['Product_Category_2'].fillna(median, inplace=True)


# In[15]:


train.Product_Category_3.value_counts()


# In[16]:


# drop Product_Category_3 
train=train.drop('Product_Category_3',axis=1)


# In[17]:


missing_values=train.isnull().sum()
percent_missing = train.isnull().sum()/train.shape[0]*100

value = {
    'missing_values':missing_values,
    'percent_missing':percent_missing
}
frame=pd.DataFrame(value)
frame


# In[18]:


#DATA VISUALISATION

train.hist(edgecolor='red',figsize=(12,12));


# In[19]:


train.columns


# In[20]:


#AGE
sns.countplot(train['Age'])
plt.title('Distribution of Age')
plt.xlabel('Different Categories of Age')
plt.show()


# In[21]:


#GENDER 
# pie chart 

size = train['Gender'].value_counts()
labels = ['Male', 'Female']
colors = ['#C4061D', 'green']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (10, 10)
plt.pie(size, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.title('A Pie Chart representing the gender gap', fontsize = 20)
plt.axis('off')
plt.legend()
plt.show()


# In[22]:


sns.countplot(x=train.Gender)
plt.title('Gender per transaction');


# In[23]:


palette=sns.color_palette("Set1")


# In[24]:


plt.rcParams['figure.figsize'] = (18, 9)
sns.countplot(train['Occupation'], palette = palette)
plt.title('Distribution of Occupation across customers', fontsize = 20)
plt.xlabel('Occupation')
plt.ylabel('Count')
plt.show()


# In[25]:


#CITY CATEGORY 
plt.rcParams['figure.figsize'] = (18, 9)
sns.countplot(train['City_Category'], palette = palette)
plt.title('Distribution of Cities across customers', fontsize = 20)
plt.xlabel('Cities')
plt.ylabel('Count')
plt.show()


# In[26]:


#PRODCTS CATEGORY
plt.figure(figsize=(20,6))
prod_by_cat = train.groupby('Product_Category_1')['Product_ID'].nunique()

sns.barplot(x=prod_by_cat.index,y=prod_by_cat.values, palette=palette)
plt.title('Number of Unique Items per Category')
plt.show()


# In[27]:


train.groupby('Product_Category_1').mean()['Purchase'].plot(kind='bar',figsize=(18,5))
plt.title("Product_Category_1 and Purchase Mean Analysis")
plt.show()


# In[28]:


# visualizing the different product categories

plt.rcParams['figure.figsize'] = (15, 25)
plt.style.use('ggplot')

plt.subplot(4, 1, 1)
sns.countplot(train['Product_Category_1'], palette = palette)
plt.title('Product Category 1', fontsize = 20)
plt.xlabel('Distribution of Product Category 1')
plt.ylabel('Count')

plt.subplot(4, 1, 2)
sns.countplot(train['Product_Category_2'], palette = palette)
plt.title('Product Category 2', fontsize = 20)
plt.xlabel('Distribution of Product Category 2')
plt.ylabel('Count')


plt.show()


# In[29]:


sns.heatmap(train.corr(),annot=True)
plt.show()


# In[30]:


#Purchase attribute which is our target variable
# importing important libraries
from scipy import stats
from scipy.stats import norm


# In[31]:


# plotting a distribution plot for the target variable
plt.rcParams['figure.figsize'] = (20, 7)
sns.distplot(train['Purchase'], color = 'green', fit = norm)

# fitting the target variable to the normal curve 
mu, sigma = norm.fit(train['Purchase']) 
print("The mu {} and Sigma {} for the curve".format(mu, sigma))

plt.title('A distribution plot to represent the distribution of Purchase')
plt.legend(['Normal Distribution ($mu$: {}, $sigma$: {}'.format(mu, sigma)], loc = 'best')
plt.show()


# In[32]:


#DATA SELECTION
train = train.drop(['Product_ID','User_ID'],axis=1)


# In[33]:


# checking the new shape of data
print(train.shape)
train


# In[34]:


#LABEL ENCODING
df_Gender = pd.get_dummies(train['Gender'])
df_Age = pd.get_dummies(train['Age'])
df_City_Category = pd.get_dummies(train['City_Category'])
df_Stay_In_Current_City_Years = pd.get_dummies(train['Stay_In_Current_City_Years'])

data_final= pd.concat([train, df_Gender, df_Age, df_City_Category, df_Stay_In_Current_City_Years], axis=1)

data_final.head()


# In[35]:


data_final = data_final.drop(['Gender','Age','City_Category','Stay_In_Current_City_Years'],axis=1)
data_final


# In[36]:


data_final.dtypes


# In[37]:


#Predicting the Amount Spent
#split data
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import accuracy_score


# In[38]:


x=data_final.drop('Purchase',axis=1)
y=data_final.Purchase


# In[39]:


print(x.shape)
print(y.shape)


# In[40]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


# In[41]:


# MODEL SELECTION
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# In[42]:


#1) LinearRegression
from sklearn.linear_model import LinearRegression


# In[43]:


lr = LinearRegression()
lr.fit(x_train,y_train)


# In[44]:


lr.intercept_


# In[45]:


lr.coef_


# In[46]:


y_pred = lr.predict(x_test)


# In[47]:


from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score


# In[48]:


mean_absolute_error(y_test, y_pred)


# In[49]:


mean_squared_error(y_test, y_pred)


# In[50]:


r2_score(y_test, y_pred)


# In[51]:


from math import sqrt
print("RMSE of Linear Regression Model is ",sqrt(mean_squared_error(y_test, y_pred)))


# In[52]:


#2)DecisionTreeRegressor

from sklearn.tree import DecisionTreeRegressor

# create a regressor object 
regressor = DecisionTreeRegressor(random_state = 0)  


# In[53]:


regressor.fit(x_train, y_train)


# In[54]:


dt_y_pred = regressor.predict(x_test)


# In[55]:


mean_absolute_error(y_test, dt_y_pred)


# In[56]:


mean_squared_error(y_test, dt_y_pred)


# In[57]:


r2_score(y_test, dt_y_pred)


# In[58]:


from math import sqrt
print("RMSE of Decision tree  Model is ",sqrt(mean_squared_error(y_test, dt_y_pred)))


# In[ ]:


#3)Random forest
from sklearn.ensemble import RandomForestRegressor

# create a regressor object 
RFregressor = RandomForestRegressor(random_state = 0) 


# In[ ]:


RFregressor.fit(x_train, y_train)


# In[ ]:


rf_y_pred = RFregressor.predict(x_test)


# In[ ]:


mean_absolute_error(y_test, rf_y_pred)


# In[ ]:


from math import sqrt
print("RMSE of Randomforest Model is ",sqrt(mean_squared_error(y_test, rf_y_pred)))


# In[ ]:


#4)XGBoost Regressor
from xgboost.sklearn import XGBRegressor


# In[ ]:


xgb_reg = XGBRegressor(learning_rate=1.0, max_depth=6, min_child_weight=40, seed=0)

xgb_reg.fit(x_train, y_train)


# In[ ]:


xgb_y_pred = xgb_reg.predict(x_test)


# In[ ]:


mean_absolute_error(y_test, xgb_y_pred)


# In[ ]:


mean_squared_error(y_test, xgb_y_pred


# In[ ]:


r2_score(y_test, xgb_y_pred)


# In[ ]:


from math import sqrt
print("RMSE ofXGBooster Regressor Model is ",sqrt(mean_squared_error(y_test, xgb_y_pred)))


# # **Among the following models we found that XGBoost classifier has the more Accuracy**

# In[69]:


score=lr.score(x_test,y_test)
print('accuracy_score overall:',score)
print('accuracy_score percent:',round(score*750,4))


# 
