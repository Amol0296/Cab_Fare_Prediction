#!/usr/bin/env python
# coding: utf-8

# In[128]:


## Importing neccessary libraries

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[129]:


#Loading the train dataset
df_train = pd.read_csv("train_cab.csv")


# In[130]:


#Load the first 10 rows of data
print(df_train.head(10))


# In[131]:


#Data type of each variable
print(df_train.dtypes)


# In[132]:


#Converting the fare amount column into numeric data form

df_train['fare_amount'] = pd.to_numeric(df_train["fare_amount"],errors="coerce")
print(df_train.dtypes)


# In[133]:


#Description of dataset
print(df_train.describe())


# ## Missing Value Analysis

# In[134]:


#Calcualte total no. of null values in train data
missing_val = pd.DataFrame(df_train.isnull().sum())
print(missing_val)


# In[135]:


#Calcuating percentage of missing values in given dataset
missing_val = missing_val.reset_index()
missing_val = missing_val.rename(columns={'index':'variables',0:'missing Value'})
missing_val['missing_percentage'] = (missing_val['missing Value']/len(df_train))*100
missing_val = missing_val.sort_values('missing_percentage',ascending=False).reset_index(drop=True)


# In[136]:


print(missing_val)


# In[137]:


#Remove a value from table and impute missing value using following method
## Fare amount
#Actual Value = 8
#Mean = 15.04
#Median = 8.5

##Passenger Count
#Actual Value = 1
#Mean = 2.62
#Median = 1

df_train['fare_amount'].loc[160] = np.nan
df_train['passenger_count'].loc[160] = np.nan


# In[138]:


print(df_train['fare_amount'].loc[160])
print(df_train['passenger_count'].loc[160])


# In[139]:


df_train['fare_amount'] = df_train['fare_amount'].fillna((df_train['fare_amount']).mean())
df_train['passenger_count'] = df_train['passenger_count'].fillna((df_train['passenger_count']).mean())


# In[140]:


print(df_train['fare_amount'].loc[160])
print(df_train['passenger_count'].loc[160])


# In[141]:


df_train['fare_amount'] = df_train['fare_amount'].fillna((df_train['fare_amount']).median())
df_train['passenger_count'] = df_train['passenger_count'].fillna((df_train['passenger_count']).median())


# In[142]:


print(df_train['fare_amount'].loc[160])
print(df_train['passenger_count'].loc[160])


# In[143]:


#Calcualte total no. of null values in train data after missing value analysis
missing_val = pd.DataFrame(df_train.isnull().sum())
print(missing_val)


# In[144]:


## Setting limit for variables

#Originally, Latitudes range from -90 to 90.
#Originally, Longitudes range from -180 to 180.
#But our data is purely negative Longitudes and purely positive latitudes
#lets align our data in its respective minimum and maximum Longitudes 
#and latitudes values, also removing fare amount,passenger count those are negative and above optimum level.

df_train = df_train[((df_train['pickup_longitude'] > -79) & (df_train['pickup_longitude'] < -70)) & 
           ((df_train['dropoff_longitude'] > -79) & (df_train['dropoff_longitude'] < -70)) & 
           ((df_train['pickup_latitude'] > 36) & (df_train['pickup_latitude'] < 45)) & 
           ((df_train['dropoff_latitude'] > 36) & (df_train['dropoff_latitude'] < 45)) & 
           ((df_train['passenger_count'] > 0) & (df_train['passenger_count'] < 7))  &
           ((df_train['fare_amount'] > 0)& (df_train['fare_amount'] < 1000))]
    


# In[145]:


#our dataset contains Datetime stamp value.we will split this data into individual columns for ease of data processing
def date_time_split(df):
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"],format='%Y-%m-%d %H:%M:%S UTC')
    df['year'] = df.pickup_datetime.dt.year
    df['month'] = df.pickup_datetime.dt.month
    df['day'] =  df.pickup_datetime.dt.day
    df['weekday'] = df.pickup_datetime.dt.weekday
    df['hour'] = df.pickup_datetime.dt.hour
    return df['pickup_datetime']
    '''
    Weekday
    0 = Monday
    1 = Tuesday
    2 = "Wednesday", 
    3= "Thursday", 
    4= "Friday", 
    5= "Saturday", 
    6= "Sunday" 
    '''
print(df_train.head())


# In[146]:


date_time_split(df_train)
#we remove datetime variable 
df_train.drop('pickup_datetime',axis=1,inplace=True)


# In[147]:


# Checking NA in data for new dataset
#Calcualte total no. of null values in train data
missing_val = pd.DataFrame(df_train.isnull().sum())
print(missing_val)


# In[148]:


#Setting proper datatypes for each variables
print(df_train.dtypes)


# In[149]:


df_train['passenger_count'] = df_train['passenger_count'].astype(int)


# In[150]:


print(df_train.dtypes)


# ## Outlier Analysis

# In[151]:


#Using Boxplot we visualize the outliers present in the data
df_train.plot(kind='box',subplots=True, layout=(8,3),sharex=False,sharey=False,fontsize=8)
plt.subplots_adjust(left=0.125,bottom=0.1,top=3,right=0.9,wspace=0.2,hspace=0.2)
plt.show()


# In[152]:


#Detecting outlier and deleting values from data

def detect_outlier(df):
    for i in df.columns:
        print(i)
        q75,q25 = np.percentile(df.loc[:,i],[75,25])
        iqr = q75-q25
        
        min = q25-(iqr*1.5)
        max = q75+(iqr*1.5)
        print(min)
        print(max)
        df = df.drop(df[df.loc[:,i] < min].index)
        df = df.drop(df[df.loc[:,i] > max].index)
    return df   


# In[153]:


df_train = detect_outlier(df_train)


# In[154]:


df_train.shape


# In[155]:


df_train.plot(kind='box',subplots=True, layout=(8,3),sharex=False,sharey=False,fontsize=8)
plt.subplots_adjust(left=0.125,bottom=0.1,top=3,right=0.9,wspace=0.2,hspace=0.2)
plt.show()


# ## Data Visualisation

# In[156]:


#Histogram plot of passaenger_count column

plt.figure(figsize=(7,7))
plt.hist(df_train['passenger_count'],bins=6)
plt.xlabel("No. of Passenger")
plt.ylabel("frequency")


# In[157]:


#Histogram plot of fare_amount column
plt.figure(figsize=(7,7))
plt.hist(df_train['fare_amount'],bins=25)
plt.xlabel("Fare Amount")
plt.ylabel("frequency")


# In[158]:


#Histogram Plot of day Column
plt.figure(figsize=(7,7))
plt.hist(df_train['day'],bins=10)
plt.xlabel('Different Days of the month')
plt.ylabel('Frequency')


# In[159]:


#Histogram Plot of weekday Column
plt.figure(figsize=(7,7))
plt.hist(df_train['weekday'],bins=10)
plt.xlabel('Different Days of the week')
plt.ylabel('Frequency')


# In[160]:


plt.figure(figsize=(7,7))
plt.hist(df_train['hour'],bins=10)
plt.xlabel('Different hours of the Day')
plt.ylabel('Frequency')


# In[161]:


#Histogram Plot of month Column
plt.figure(figsize=(7,7))
plt.hist(df_train['month'],bins=10)
plt.xlabel('Different Months of the year')
plt.ylabel('Frequency')


# In[162]:


#Histogram Plot of year Column
plt.figure(figsize=(7,7))
plt.hist(df_train['year'],bins=10)
plt.xlabel('Years')
plt.ylabel('Frequency')


# In[163]:


#Histogram Plot of dropoff_latitude Column
plt.figure(figsize=(7,7))
plt.hist(df_train['dropoff_latitude'])
plt.xlabel('dropoff_latitude')
plt.ylabel('Frequency')


# In[164]:


#Histogram Plot of dropoff_longitude Column
plt.figure(figsize=(7,7))
plt.hist(df_train['dropoff_longitude'])
plt.xlabel('dropoff_longitude')
plt.ylabel('Frequency')


# In[165]:


#Histogram Plot of pickup_latitude Column
plt.figure(figsize=(7,7))
plt.hist(df_train['pickup_latitude'])
plt.xlabel('pickup_latitude')
plt.ylabel('Frequency')


# In[166]:


#################Scatter plot########################

fig,x = plt.subplots(nrows=6,ncols=2)
fig.set_size_inches(12,15)

sns.scatterplot(x="passenger_count", y="fare_amount", data= df_train, palette="Set2",ax=x[0][0])
sns.scatterplot(x="month", y="fare_amount", data= df_train, palette="Set2",ax=x[0][1])
sns.scatterplot(x="weekday", y="fare_amount", data= df_train, palette="Set2",ax=x[1][0])
sns.scatterplot(x="hour", y="fare_amount", data= df_train, palette="Set2",ax=x[1][1])
sns.scatterplot(x="pickup_longitude", y="fare_amount", data= df_train, palette="Set2",ax=x[2][0])
sns.scatterplot(x="pickup_latitude", y="fare_amount", data= df_train, palette="Set2",ax=x[2][1])
sns.scatterplot(x="pickup_latitude", y="pickup_longitude", data= df_train, palette="Set2",ax=x[3][0])
sns.scatterplot(x="dropoff_latitude", y="dropoff_longitude", data= df_train, palette="Set2",ax=x[3][1])
sns.scatterplot(x="dropoff_longitude", y="fare_amount", data= df_train, palette="Set2",ax=x[4][0])
sns.scatterplot(x="dropoff_latitude", y="fare_amount", data= df_train, palette="Set2",ax=x[4][1])
sns.scatterplot(x="day", y="fare_amount", data= df_train, palette="Set2",ax=x[5][0])


# ## Feature Selection

# In[167]:


#Correlation Analysis

def Correlation(df):
    df_corr = df.loc[:,df.columns]
    corr = df_corr.corr()
    sns.set()
    plt.figure(figsize=(9,9))
    sns.heatmap(corr,annot= True,fmt = ".3f",square=True,linewidths = 0.5)
    
Correlation(df_train)


# In[168]:


######Splitting Dataset into train and test #########

train,test = train_test_split(df_train,test_size=0.2,random_state = 121
                             )
X_train = train.iloc[:,1:11]
Y_train = train.iloc[:,0]
X_test = test.iloc[:,1:11]
Y_test = test.iloc[:,0]
print(X_train.head())


# ## Feature Scaling

# In[169]:


#Normalisation

def Normalisation(df):
    for i in df.columns:
        print(i)
        df[i] = (df[i] - df[i].min())/(df[i].max() - df[i].min())

Normalisation(X_train)
Normalisation(X_test)
print(X_train.head(10))


# ## Model Developement

# ## Defining Error Metrics

# In[170]:


#MAPE

def MAPE(y_true, y_pred):
    MAE = np.mean(np.abs((y_true - y_pred)))
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    print("MAE is:", MAE)
    print("MAPE is:", mape)
    return mape


# In[171]:


#RMSE

def RMSE(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    print("MSE: ",mse)
    print("RMSE: ",rmse)
    return rmse
    


# ## Linear Regression

# In[172]:




LR_model = sm.OLS(train.iloc[:,0],train.iloc[:,1:]).fit()


# In[173]:


print(LR_model.summary())


# In[174]:


#Prediction
LR_model_predict = LR_model.predict(test.iloc[:,1:])


# In[175]:


#Model Evaluation

# df_Result['MAPE'] = 
# df_Result['RMSE'] = 

MAPE(Y_test,LR_model_predict)
RMSE(Y_test,LR_model_predict)
# MAE is: 2.973392366538636
# MAPE is: 0.41842477657562493
# MSE:  14.442326262657845
# RMSE:  3.800306074865266


# ## KNN 

# In[176]:


KNN_model = KNeighborsRegressor(n_neighbors=50).fit(X_train,Y_train)

#Predict
KNN_model_predict = KNN_model.predict(X_test)


# In[177]:


#Model Evaluation
MAPE(Y_test,KNN_model_predict)
RMSE(Y_test,KNN_model_predict)

# MAE is: 2.452239522175146
# MAPE is: 0.3198996671035521
# MSE:  10.642973863374007
# RMSE:  3.262357102368471


# ## Decision Tree

# In[178]:



DT_model = DecisionTreeRegressor(random_state=123).fit(train.iloc[:,1:],train.iloc[:,0])
DT_model_predict = DT_model.predict(test.iloc[:,1:])


# In[179]:


#Model Evaluation
MAPE(Y_test,DT_model_predict)
RMSE(Y_test,DT_model_predict)


# MAE is: 2.279578587009114
# MAPE is: 0.2959167425706746
# MSE:  9.514700357563429
# RMSE:  3.0845907925628366


# ## Random Forest
# 

# In[180]:


RF_model = RandomForestRegressor(max_features='auto',n_estimators=500,max_depth = 8).fit(train.iloc[:,1:],train.iloc[:,0])
RF_model_predict = RF_model.predict(test.iloc[:,1:])


# In[181]:


#Model Evaluation
MAPE(Y_test,RF_model_predict)
RMSE(Y_test,RF_model_predict)

# MAE is: 2.1770230876697245
# MAPE is: 0.31314930719311923
# MSE:  8.012735122578105
# RMSE:  2.8306775023972803


# In[182]:


df_Result = pd.DataFrame({'Model':["Linear Regression","KNN","Decision Tree","Random Forest"],
                           'MAPE':["0.41842477657562493","0.3198996671035521","0.2959167425706746","0.31314930719311923"],
                            'RMSE':["3.800306074865266","3.262357102368471","3.0845907925628366","2.8306775023972803"]})

df_Result = df_Result[['Model','MAPE','RMSE']]
df_Result.sort_values('RMSE',ascending =True)


# In[183]:


##Random Forest has lowest RMSE Value than other model.
## So we will freeze Random forest model for our prediction on test data


# ## Testing of Model on test data

# In[184]:


#Loading test dataset
df_test = pd.read_csv("test.csv")


# In[185]:


df_test.describe()


# In[186]:


#After checking test data we limit data into a range
df_test = df_test[((df_test['pickup_longitude'] > -79) & (df_test['pickup_longitude'] < -70)) & 
           ((df_test['dropoff_longitude'] > -79) & (df_test['dropoff_longitude'] < -70)) & 
           ((df_test['pickup_latitude'] > 36) & (df_test['pickup_latitude'] < 45)) & 
           ((df_test['dropoff_latitude'] > 36) & (df_test['dropoff_latitude'] < 45)) & 
           (df_test['passenger_count'] > 0) ]


# In[187]:


## splitting datetime data into different columns and removing pickup_datetime column
date_time_split(df_test)
df_test.drop('pickup_datetime',axis= 1,inplace=True)
print(df_test.isnull().sum())


# In[188]:


df_test.dtypes


# In[189]:


## Using previously trained model of random forest for prediction on test data
rf_trained_model = RF_model.predict(df_test)


# In[190]:


rf_trained_model = pd.DataFrame(rf_trained_model,columns = ["fare_amount"])


# In[191]:


##saving the result into local machine
rf_trained_model.to_csv("Test_data_predictions.csv",index=False)


# In[ ]:




