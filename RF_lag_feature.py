# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 19:24:36 2023

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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.model_selection import TimeSeriesSplit




def kge_metric(simulated, observed):
    """Calculate Kling-Gupta Efficiency (KGE)."""
    mean_sim = np.mean(simulated)
    mean_obs = np.mean(observed)

    std_sim = np.std(simulated)
    std_obs = np.std(observed)

    correlation = np.corrcoef(simulated, observed)[0, 1]

    kge = 1 - np.sqrt((correlation - 1) ** 2 + (std_sim / std_obs - 1) ** 2 + (mean_sim / mean_obs - 1) ** 2)

    return kge









input_file_path = r'C:\Users\pmoghaddasi\Desktop\Snow\proposal_basins\merged_hysets_daymet_GLDAS_AMSR_snowdas_UAZ_MERRA2_hysets_09112200.csv'
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
mse_values = {}
kge_values = {}


# Loop through each feature
for feature in features:

    # Loop through each lag
    for lag in lags:
        # Create lagged features for the current feature
        lagged_feature = f'{feature}_{lag}_lag'
        lagged_features = [lagged_feature]

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
        
        # Normalize the data
        mean_train_X = X_train.mean()
        mean_train_y = y_train.mean()
    
        std_train_X = X_train.std()
        std_train_y = y_train.std()
        
        
        X_train = (X_train - mean_train_X)/ std_train_X
        X_test = (X_test - mean_train_X)/ std_train_X
        
        y_train = (y_train - mean_train_y)/ std_train_y
        y_test = (y_test - mean_train_y)/ std_train_y
        
        
        forest_para = {'n_estimators':[10,20,50,75,100,200,300], 'max_depth':[2,3,4,5,7,10],'min_samples_leaf':[2,4,6,8,10], 'random_state':[42]}
        
                
        forest_reg = RandomForestRegressor()
        
        
        # grid_search = GridSearchCV(forest_reg, forest_para, cv=5, scoring='neg_mean_squared_error',
        # return_train_score=True)
        
        
        # grid_search.fit(X_train, y_train)
        
        # Define time series cross-validator
        # tscv = TimeSeriesSplit(n_splits=5)
        
        # grid_search = GridSearchCV(forest_reg, forest_para, cv=tscv.split(X_train), scoring='neg_mean_squared_error', return_train_score=True)
        grid_search = GridSearchCV(forest_reg, forest_para, cv=5, scoring='neg_mean_squared_error', return_train_score=True)

        grid_search.fit(X_train, y_train)
   
        print(grid_search.best_params_)
        
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        
        
        r2  = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        kge = kge_metric(y_pred, y_test)
        
        # Store metric value in the dictionary
        r2_values [(feature, lag)] = r2 
        mse_values[(feature, lag)] = mse
        kge_values[(feature, lag)] = kge

        print(f"For {feature} with lag {lag}, R-squared: {r2}")
        
print("R-squared values:", r2_values )
print("Mean Squared Error (MSE) values:", mse_values)
print("Kling-Gupta Efficiency (KGE) values:", kge_values)


# Extract feature names and lags for labeling
feature_labels = [feature for feature in features for lag in lags]
lag_labels = [lag for feature in features for lag in lags]

# Create a list of colors for better visualization
colors = ['b', 'g', 'r', 'c', 'm', 'y']

# Plot R-squared values
plt.figure(figsize=(12, 6))
for i, (key, value) in enumerate(r2_values.items()):
    feature, lag = key
    plt.bar(i, value, color=colors[i % len(colors)], label=f'{feature} with lag {lag}')

# Set x-axis labels and ticks
plt.xticks(range(len(r2_values)), [f'{feature_labels[i]}\n(Lag {lag_labels[i]})' for i in range(len(r2_values))], rotation=45, ha='right')

# Set y-axis label
plt.ylabel('R-squared Value')

# # Add legend
# plt.legend()

# Set plot title
plt.title('R-squared Values for Different Feature-Lag Combinations')

# Show the plot
plt.tight_layout()
plt.show()






