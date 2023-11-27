# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:59:36 2023

@author: pmoghaddasi
"""

import pandas as pd
import numpy as np
from numpy import save
from numpy import load
import os
import datetime
from scipy.stats import rankdata
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.utils import shuffle




s=time.time()

input_file_path = 'merged_hysets_daymet_GLDAS_AMSR_snowdas_UAZ_MERRA2_hysets_09112200.csv'
# Read the CSV file
df = pd.read_csv(input_file_path)

output_file_name =input_file_path.split('_')[-1].split('.')[0]


# Select the desired columns
desired_columns = ['date', 'snow_depth_water_equivalent_mean', 'streamflow', 'swe_daymet','swe_UAZ','swe_GLDAS']

#desired_columns = ['date', 'snow_depth_water_equivalent_mean', 'surface_net_solar_radiation_mean', 'temperature_2m_mean', 'total_precipitation_sum', 'streamflow', 'swe_daymet', 'swe_GLDAS', 'swe_AMSR', 'swe_snowdas', 'swe_UAZ', 'swe_MERRA2']

df = df[desired_columns]

df.dropna(inplace=True)

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')


# Set the 'date' column as the index
df.set_index('date', inplace=True)


columns_to_average = ['snow_depth_water_equivalent_mean', 'streamflow', 'swe_daymet','swe_UAZ','swe_GLDAS']

# Define the list of lags for the moving averages
lags = [0, 7, 14, 30, 60, 91]  


# Calculate the moving averages for each specified column and each window size
for column in columns_to_average:
    for lag in lags:
        df[f'{column}_{lag}_lag'] = df[column].shift(lag)

df.dropna(inplace=True)


# Define the features and target variable
features = ['swe_daymet','swe_UAZ','swe_GLDAS', 'snow_depth_water_equivalent_mean']



# Initialize an empty dictionary to store MSE values
r2_values = {}

# Loop through each lag
for lag in lags:
    # Create lagged features for the current feature
    lagged_features = [f'{feature}_{lag}_lag' for feature in features]

    # Define the target variable
    target = 'streamflow'

    # Split the data into training and testing sets
    X = df[lagged_features]
    y = df[target]
    
    n = len(X)    
    X_train = X[:int(n*0.85)]
    X_test = X[int(n*0.85):]
    y_train = y[:int(n*0.85)]
    y_test = y[int(n*0.85):]
    X_train, y_train= shuffle(X_train, y_train, random_state=42)
    
    ##TODO: do this with pandas
    mean_train_X = np.mean(X_train)
    mean_train_y = np.mean(y_train)

    std_train_X = np.std(X_train)
    std_train_y = np.std(y_train)    
    
    X_train = (X_train - mean_train_X)/ std_train_X
    X_test = (X_test - mean_train_X)/ std_train_X
    
    y_train = (y_train - mean_train_y)/ std_train_y
    y_test = (y_test - mean_train_y)/ std_train_y    

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
    #X_train, X_val, y_train, y_val = train_test_split(X_train1, y_train1, test_size=0.15, random_state=42)
    
    
    forest_para = {'n_estimators':[10,20,50,75,100,200,300], 'max_depth':[2,3,4,5,7,10],'min_samples_leaf':[2,4,6,8,10], 'random_state':[42]}
    
            
    forest_reg = RandomForestRegressor()
    
    
    grid_search = GridSearchCV(forest_reg, forest_para, cv=5, scoring='neg_mean_squared_error',
    return_train_score=True)
    
    
    grid_search.fit(X_train, y_train)
    
    print(grid_search.best_params_)
    
    best_model = grid_search.best_estimator_
    
    y_pred = best_model.predict(X_test)
    
    ##TODO: add other performance metrics
    r2  = r2_score(y_test, y_pred)
    
    # Store MSE value in the dictionary
    r2_values [lag] = r2 

    print(f"For {lag}, R-squared: {r2}")
        
print("R-squared values:", r2_values )

t=time.time()
print("run time", t-s)