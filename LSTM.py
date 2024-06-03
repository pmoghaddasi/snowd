# -*- coding: utf-8 -*-
"""
Created on Fri May 31 19:25:14 2024

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
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Read the CSV file
input_file_path = r'C:\Users\pmoghaddasi\Desktop\01052500_merged_data.csv'
df = pd.read_csv(input_file_path)

# Select the desired columns
# desired_columns = ['date', 'daymet_swe', 'uaz_swe', 'snodas_swe', 'OBS_RUN']
desired_columns = ['date', 'uaz_swe', 'OBS_RUN']
df = df[desired_columns]


# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')

# Set the 'date' column as the index
df.set_index('date', inplace=True)


# Replace -999.0 values with NaN to handle them properly
df.replace(-999.0, pd.NA, inplace=True)

# Drop NaN values
df.dropna(inplace=True)


# Define the features and target variable
features = ['uaz_swe']

# Define the list of lags for the moving averages
lags = [7, 14, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330, 364]
# lags = [7 * i for i in range(1, 53)]
lags = [7, 14, 30, 45, 60]

# Function to create sequences for LSTM
def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data.iloc[i:i + n_past, :-1])  # Select all features except the last column (target)
        y.append(data.iloc[i + n_past:i + n_past + n_future, -1])  # Select the last column as the target
    return np.array(X), np.array(y)



# Function to calculate Nash-Sutcliffe Efficiency
def nash_sutcliffe(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)


def kge_metric(simulated, observed):
    """Calculate Kling-Gupta Efficiency (KGE)."""
    mean_sim = np.mean(simulated)
    mean_obs = np.mean(observed)

    std_sim = np.std(simulated)
    std_obs = np.std(observed)

    correlation = np.corrcoef(simulated, observed)[0, 1]

    kge = 1 - np.sqrt((correlation - 1) ** 2 + (std_sim / std_obs - 1) ** 2 + (mean_sim / mean_obs - 1) ** 2)

    return kge


# Initialize empty dictionaries to store performance metrics
r2_values_train = {}
mse_values_train = {}
nse_values_train = {}

r2_values_val = {}
mse_values_val = {}
nse_values_val = {}

r2_values_test = {}
mse_values_test = {}
nse_values_test = {}


# Loop through each lag
for lag in lags:

    # Define the target variable
    target_column = 'OBS_RUN'
    X_seq, y_seq = create_sequences(df[[f'{feature}' for feature in features]+[target_column]],
                             n_past=lag, n_future=1)
    
    
    # Train-test split
    split_point = int(0.85 * len(X_seq))
    X_train, X_test = X_seq[:split_point], X_seq[split_point:]
    y_train, y_test = y_seq[:split_point], y_seq[split_point:]
    
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    
    
    # Normalize the data
    mean_train_X = X_train.mean()
    mean_train_y = y_train.mean()
    
    std_train_X = X_train.std()
    std_train_y = y_train.std()
    
    X_train = (X_train - mean_train_X) / std_train_X
    X_val = (X_val - mean_train_X) / std_train_X
    X_test = (X_test - mean_train_X) / std_train_X
    y_train = (y_train - mean_train_y) / std_train_y
    y_val = (y_val - mean_train_y) / std_train_y
    y_test = (y_test - mean_train_y) / std_train_y

    
    # Build and train the LSTM model
    model = Sequential()
    model.add(LSTM(units=30, return_sequences=True, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dropout(0.2))
    model.add(LSTM(units=20, return_sequences=True, activation='relu'))
    model.add(Dropout(0.2))
    model.add(LSTM(units=10, return_sequences=False, activation='relu'))
    model.add(Dense(units=1))
    
    
    optimizer = Adam(learning_rate=0.01)
    model.compile(optimizer=optimizer, loss='mse')
    # model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=40, batch_size=64, validation_data=(X_val, y_val))
    
    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Calculate performance metrics for the training set
    r2_train = r2_score(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    nse_train = nash_sutcliffe(y_train, y_pred_train)

    # Calculate performance metrics for the validation set
    r2_val = r2_score(y_val, y_pred_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    nse_val = nash_sutcliffe(y_val, y_pred_val)

    # Calculate performance metrics for the test set
    r2_test = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    nse_test = nash_sutcliffe(y_test, y_pred_test)

    # Store metric values in the dictionaries
    r2_values_train[lag] = r2_train
    mse_values_train[lag] = mse_train
    nse_values_train[lag] = nse_train

    r2_values_val[lag] = r2_val
    mse_values_val[lag] = mse_val
    nse_values_val[lag] = nse_val

    r2_values_test[lag] = r2_test
    mse_values_test[lag] = mse_test
    nse_values_test[lag] = nse_test

# =============================================================================
#     print(f"For {lag}, R-squared: {r2_test}")
# 
#     # Plot observed vs predicted for the training set
#     plt.figure(figsize=(14, 6))
#     plt.plot(y_train, label='Observed', color='blue')
#     plt.plot(y_pred_train, label='Predicted', color='red', linestyle='--')
#     plt.title(f'Observed vs Predicted for Training Set (Lag {lag})')
#     plt.xlabel('Time')
#     plt.ylabel('Streamflow')
#     plt.legend()
#     plt.show()
# 
#     # Plot observed vs predicted for the validation set
#     plt.figure(figsize=(14, 6))
#     plt.plot(y_val, label='Observed', color='blue')
#     plt.plot(y_pred_val, label='Predicted', color='red', linestyle='--')
#     plt.title(f'Observed vs Predicted for Validation Set (Lag {lag})')
#     plt.xlabel('Time')
#     plt.ylabel('Streamflow')
#     plt.legend()
#     plt.show()
# 
# =============================================================================
    # Plot observed vs predicted for the test set
    plt.figure(figsize=(14, 6))
    plt.plot(y_test, label='Observed', color='blue')
    plt.plot(y_pred_test, label='Predicted', color='red', linestyle='--')
    plt.title(f'Observed vs Predicted for Test Set (Lag {lag})')
    plt.xlabel('Time')
    plt.ylabel('Streamflow')
    plt.legend()
    plt.show()

# =============================================================================
# print("R-squared values for Training Set:", r2_values_train)
# print("Mean Squared Error (MSE) values for Training Set:", mse_values_train)
# print("Nash-Sutcliffe Efficiency (NSE) values for Training Set:", nse_values_train)
# 
# print("R-squared values for Validation Set:", r2_values_val)
# print("Mean Squared Error (MSE) values for Validation Set:", mse_values_val)
# print("Nash-Sutcliffe Efficiency (NSE) values for Validation Set:", nse_values_val)
# 
# print("R-squared values for Test Set:", r2_values_test)
# print("Mean Squared Error (MSE) values for Test Set:", mse_values_test)
# print("Nash-Sutcliffe Efficiency (NSE) values for Test Set:", nse_values_test)
# 
# 
# =============================================================================
    
    # Create a DataFrame to store the performance metrics
    metrics_df = pd.DataFrame({
        'Metric': ['R2', 'MSE', 'NSE'],
        'Train': [r2_values_train[lag], mse_values_train[lag], nse_values_train[lag]],
        'Validation': [r2_values_val[lag], mse_values_val[lag], nse_values_val[lag]],
        'Test': [r2_values_test[lag], mse_values_test[lag], nse_values_test[lag]],
    })
    
    # Transpose the DataFrame for the desired format
    metrics_df = metrics_df.set_index('Metric').transpose()
    
    # Display the table
    print(metrics_df)
    
    
    
    
    
# Plot R-squared values for all lags in one plot
plt.figure(figsize=(10, 6))
plt.plot(lags, [r2_values_train[lag] for lag in lags], marker='o', label='Train R2')
plt.plot(lags, [r2_values_val[lag] for lag in lags], marker='o', label='Validation R2')
plt.plot(lags, [r2_values_test[lag] for lag in lags], marker='o', label='Test R2')
plt.xlabel('Lag (days)')
plt.ylabel('R-squared')
plt.title('R-squared Values for Different Lags')
plt.legend()
plt.grid(True)
plt.show()

print("R-squared values for Training Set:", r2_values_train)
print("Mean Squared Error (MSE) values for Training Set:", mse_values_train)
print("Nash-Sutcliffe Efficiency (NSE) values for Training Set:", nse_values_train)

print("R-squared values for Validation Set:", r2_values_val)
print("Mean Squared Error (MSE) values for Validation Set:", mse_values_val)
print("Nash-Sutcliffe Efficiency (NSE) values for Validation Set:", nse_values_val)

print("R-squared values for Test Set:", r2_values_test)
print("Mean Squared Error (MSE) values for Test Set:", mse_values_test)
print("Nash-Sutcliffe Efficiency (NSE) values for Test Set:", nse_values_test)