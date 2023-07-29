# import libraries
import numpy as np #useful for numpy arrays
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv
import matplotlib.pyplot as plt # for making plots
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split  #to split data into train and test
from xgboost import XGBRegressor # this is good for classification, predict the categories 
from sklearn import metrics # gives some metrics functions to find the performance of our model 

# Data collection, analyses and processing
# load the data from csv to pandas dataFrame
big_mart_data = pd.read_csv('Big Mart Prediction\Train.csv') 
# pandas dataFrame is a more structured dataframe

#print the row of the dataFrame
big_mart_data.head()

# number of data points & number features
#data points represent rowns & features represents columns
big_mart_data.shape

#get information about the dataset(what are the various data types)
big_mart_data.info()

#what are the categorical features
# Item_Identifier
# Item_Fat_Content
# Item_Type
# Outlet_Identifier
# Outlet_Size
# Outlet_Location_Type
# Outlet_Type

#perform analysis on the categorical features before tsaken it to the ML Module
# checking for missing values
big_mart_data.isnull().sum()

## handle the missing values
# we will need to metrics(mean --> average value, mode --> most repeated value)
# convert all the missing values using mean. 

#what is the mean value of "Item_Weight Column"
big_mart_data['Item_Weight'].mean()

#fill the particular column with mean value
big_mart_data["Item_Weight"].fillna(big_mart_data['Item_Weight'].mean(), inplace = True)

# checking for missing values of Item_Weight Column
big_mart_data.isnull().sum()

#Replace the missing value in Outlet_Size Column with Mode
mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))

print(mode_of_outlet_size)
# it'll take all the missing values and compare it with the outlet_type and replace it, depending on the outlet_type

missing_values = big_mart_data['Outlet_Size'].isnull()
print(missing_values)

#replace the missing values with the mode
big_mart_data.loc[missing_values, "Outlet_Size"] = big_mart_data.loc[missing_values, "Outlet_Type"].apply(lambda x: mode_of_outlet_size)

# checking for missing values
big_mart_data.isnull().sum()

#perform analysis on the data
# put it in a graph
big_mart_data.describe()

#Numerical Features
sns.set()

# how the value is distributed in this column
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Weight'])
plt.show()

#item visibility
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Visibility'])
plt.show()

#item mrp
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_MRP'])
plt.show()

#item OUTLET SALES DISTRIBUTION
plt.figure(figsize=(6,6))
sns.distplot(big_mart_data['Item_Outlet_Sales'])
plt.show()

#outlet_establishment_year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=big_mart_data)
plt.show()

#check categorical Features
#Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=big_mart_data)
plt.show()

#clean the column
plt.figure(figsize=(20,6))
sns.countplot(x='Item_Type', data=big_mart_data)
plt.show()

#plot the item sales column
#---------------------------------------#
#plt.figure(figsize=(6,6))
#sns.countplot(x='Outlet_Size', data=big_mart_data)
#plt.show()
#--------------------------------------#

# Convert 'Outlet_Size' column to string data type
big_mart_data['Outlet_Size'] = big_mart_data['Outlet_Size'].astype(str)

# Check unique values in 'Outlet_Size' column
print(big_mart_data['Outlet_Size'].unique())

# Convert 'Outlet_Size' column to categorical data type
big_mart_data['Outlet_Size'] = pd.Categorical(big_mart_data['Outlet_Size'])

# Plot the countplot
plt.figure(figsize=(6, 6))
sns.countplot(x='Outlet_Size', data=big_mart_data)
plt.title('Item_Type count')
plt.show()

#data preprocessing
big_mart_data.head()

big_mart_data['Item_Fat_Content'].value_counts()

big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg':'Regular'}}, inplace=True)

big_mart_data['Item_Fat_Content'].value_counts()

#trnasform all categorical values into
#Label Encoding
encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

big_mart_data.head()

# Splitting Features & Targets
X = big_mart_data.drop(columns=['Item_Outlet_Sales'], axis=1) #if removing a column, axis=1 but if row axis=0 (features) 
Y = big_mart_data['Item_Outlet_Sales'] #(target)
print(X)
print(Y)

# Split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2) #0.2 means taking 20% of the dataset/ random_state is like an identification number

print(X.shape, X_train.shape, X_test.shape)

# Function to calculate Mean Squared Error (MSE)
def calculate_mse(actual_values, predicted_values):
    return np.mean((actual_values - predicted_values) ** 2)

# Function to calculate Mean Absolute Error (MAE)
def calculate_mae(actual_values, predicted_values):
    return np.mean(np.abs(actual_values - predicted_values))


# Machine Learning Model Training
#XGBoost Regressor
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)

# Evaluation
## prediction on training data
training_data_prediction = regressor.predict(X_train)

# R square value is good for checking the performance of our value
r2_train = metrics.r2_score(Y_train, training_data_prediction)

print('R Squared Value (Train) = ', r2_train) # if it is closer to 1 then it is a good model

#predict for testing data
test_data_prediction = regressor.predict(X_test)

r2_test = metrics.r2_score(Y_test, test_data_prediction)

print('R Squared Value (Test) = ', r2_test) # if it is closer to 1 then it is a good model

# Calculate Mean Squared Error (MSE) for training and testing data
mse_train = calculate_mse(Y_train, training_data_prediction)
mse_test = calculate_mse(Y_test, test_data_prediction)

print('Mean Squared Error (Train) = ', mse_train)
print('Mean Squared Error (Test) = ', mse_test)

# Calculate Mean Absolute Error (MAE) for training and testing data
mae_train = calculate_mae(Y_train, training_data_prediction)
mae_test = calculate_mae(Y_test, test_data_prediction)

print('Mean Absolute Error (Train) = ', mae_train)
print('Mean Absolute Error (Test) = ', mae_test)