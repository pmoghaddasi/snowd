# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 16:26:56 2023

@author: pmoghaddasi
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def kge_metric(simulated, observed):
    """Calculate Kling-Gupta Efficiency (KGE)."""
    mean_sim = np.mean(simulated)
    mean_obs = np.mean(observed)

    std_sim = np.std(simulated)
    std_obs = np.std(observed)

    correlation = np.corrcoef(simulated, observed)[0, 1]

    kge = 1 - np.sqrt((correlation - 1) ** 2 + (std_sim / std_obs - 1) ** 2 + (mean_sim / mean_obs - 1) ** 2)

    return kge




# Read the CSV file
input_file_path = r'C:\Users\pmoghaddasi\Desktop\Snow\proposal_basins\merged_hysets_daymet_GLDAS_AMSR_snowdas_UAZ_MERRA2_hysets_09112200.csv'
df = pd.read_csv(input_file_path)

# Select the desired columns
desired_columns = ['date', 'snow_depth_water_equivalent_mean', 'streamflow', 'swe_daymet', 'swe_UAZ', 'swe_GLDAS']
df = df[desired_columns]

# Drop NaN values
df.dropna(inplace=True)

# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Define the features and target variable
features = ['swe_daymet', 'swe_UAZ', 'swe_GLDAS', 'snow_depth_water_equivalent_mean']

# Define the list of lags for the moving averages
lags = [0, 7, 14, 30, 60, 91]

# Calculate the moving averages for each specified column and each window size
for column in features:
    for lag in lags:
        df[f'{column}_{lag}_lag'] = df[column].shift(lag)

df.dropna(inplace=True)

# Initialize an empty dictionary to store performance metrics
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
        X_train = X[:int(n * 0.85)]
        X_test = X[int(n * 0.85):]
        y_train = y[:int(n * 0.85)]
        y_test = y[int(n * 0.85):]
        X_train, y_train = shuffle(X_train, y_train, random_state=42)

        # Normalize the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()

        X_train = scaler_X.fit_transform(X_train)
        X_test = scaler_X.transform(X_test)

        y_train = scaler_y.fit_transform(y_train.values.reshape(-1, 1))
        y_test = scaler_y.transform(y_test.values.reshape(-1, 1))

        # Reshape input data for LSTM
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        # Build LSTM model
        model = Sequential()
        model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=1, activation='linear'))
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

        # Predictions
        y_pred = model.predict(X_test)

        # Inverse transform predictions to original scale
        y_pred = scaler_y.inverse_transform(y_pred)
        y_test = scaler_y.inverse_transform(y_test)

        # Calculate performance metrics
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        kge = kge_metric(y_pred.flatten(), y_test.flatten())

        # Store metric values in the dictionary
        r2_values[(feature, lag)] = r2
        mse_values[(feature, lag)] = mse
        kge_values[(feature, lag)] = kge

        print(f"For {feature} with lag {lag}, R-squared: {r2}")

# Print final results
print("R-squared values:", r2_values)
print("Mean Squared Error (MSE) values:", mse_values)
print("Kling-Gupta Efficiency (KGE) values:", kge_values)
