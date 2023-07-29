# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

# Data collection, analyses, and processing
# Load the data from csv to pandas dataFrame
big_mart_data = pd.read_csv('Big Mart Prediction\Train.csv')

#print the row of the dataFrame
big_mart_data.head()

# Handle missing values
big_mart_data["Item_Weight"].fillna(big_mart_data['Item_Weight'].mean(), inplace=True)

mode_of_outlet_size = big_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
missing_values = big_mart_data['Outlet_Size'].isnull()
big_mart_data.loc[missing_values, "Outlet_Size"] = big_mart_data.loc[missing_values, "Outlet_Type"].apply(lambda x: mode_of_outlet_size)

# Convert 'Outlet_Size' column to string data type
big_mart_data['Outlet_Size'] = big_mart_data['Outlet_Size'].astype(str)

# Data preprocessing
big_mart_data.replace({'Item_Fat_Content': {'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'}}, inplace=True)
encoder = LabelEncoder()
big_mart_data['Item_Identifier'] = encoder.fit_transform(big_mart_data['Item_Identifier'])
big_mart_data['Item_Fat_Content'] = encoder.fit_transform(big_mart_data['Item_Fat_Content'])
big_mart_data['Item_Type'] = encoder.fit_transform(big_mart_data['Item_Type'])
big_mart_data['Outlet_Identifier'] = encoder.fit_transform(big_mart_data['Outlet_Identifier'])
big_mart_data['Outlet_Size'] = encoder.fit_transform(big_mart_data['Outlet_Size'])
big_mart_data['Outlet_Location_Type'] = encoder.fit_transform(big_mart_data['Outlet_Location_Type'])
big_mart_data['Outlet_Type'] = encoder.fit_transform(big_mart_data['Outlet_Type'])

# Data preprocessing
categorical_columns = big_mart_data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    encoder = LabelEncoder()
    big_mart_data[column] = encoder.fit_transform(big_mart_data[column])

# Splitting Features & Targets
X = big_mart_data.drop(columns=['Item_Outlet_Sales'], axis=1)
Y = big_mart_data['Item_Outlet_Sales']

# Split data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Machine Learning Model Training
# Random Forest Regressor
regressor = RandomForestRegressor()
regressor.fit(X_train, Y_train)

# Evaluation
# Prediction on training data
training_data_prediction = regressor.predict(X_train)
r2_train = metrics.r2_score(Y_train, training_data_prediction)
mse_train = metrics.mean_squared_error(Y_train, training_data_prediction)
mae_train = metrics.mean_absolute_error(Y_train, training_data_prediction)

print('R Squared Value (Train) = ', r2_train)
print('Mean Squared Error (Train) = ', mse_train)
print('Mean Absolute Error (Train) = ', mae_train)

# Predict for testing data
test_data_prediction = regressor.predict(X_test)
r2_test = metrics.r2_score(Y_test, test_data_prediction)
mse_test = metrics.mean_squared_error(Y_test, test_data_prediction)
mae_test = metrics.mean_absolute_error(Y_test, test_data_prediction)

print('R Squared Value (Test) = ', r2_test)
print('Mean Squared Error (Test) = ', mse_test)
print('Mean Absolute Error (Test) = ', mae_test)
