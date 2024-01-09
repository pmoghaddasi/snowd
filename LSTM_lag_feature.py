# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 15:52:12 2024

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
input_file_path = 'merged_hysets_daymet_GLDAS_AMSR_snowdas_UAZ_MERRA2_hysets_09112200.csv'
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


# Function to create sequences for LSTM
def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data.iloc[i:i + n_past, 0])
        y.append(data.iloc[i + n_past:i + n_past + n_future, 1])
    return np.array(X), np.array(y)






# Initialize an empty dictionary to store performance metrics
r2_values = {}
mse_values = {}
kge_values = {}

# Loop through each feature
for feature in features:
    # Loop through each lag
    for lag in lags:

        lag = 5
        # Define the target variable
        target_column  = 'streamflow'
        

        X_seq, y_seq = create_sequences(df[[f'{feature}_0_lag'] + [target_column]],
                                 n_past=lag+1, n_future=1)
        
        # Train-test split
        split_point = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_point], X_seq[split_point:]
        y_train, y_test = y_seq[:split_point], y_seq[split_point:]
        
        

        # Normalize the data
        mean_train_X = X_train.mean()
        mean_train_y = y_train.mean()
        
        std_train_X = X_train.std()
        std_train_y = y_train.std()
        
        X_train = (X_train - mean_train_X) / std_train_X
        X_test = (X_test - mean_train_X) / std_train_X
        
        y_train = (y_train - mean_train_y) / std_train_y
        y_test = (y_test - mean_train_y) / std_train_y

        
        # Build and train the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], 50)))
        model.add(Dense(units=10))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2)
        
        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}')
        
        # Make predictions
        y_pred = model.predict(X_test)
        

        
        # Calculate R2 and MSE
        r2  = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        kge = kge_metric(y_pred, y_test)

        # Store metric values in the dictionary
        r2_values[(feature, lag)] = r2
        mse_values[(feature, lag)] = mse
        kge_values[(feature, lag)] = kge

        print(f"For {feature} with lag {lag}, R-squared: {r2}")
        aa

# Print final results
print("R-squared values:", r2_values)
print("Mean Squared Error (MSE) values:", mse_values)
print("Kling-Gupta Efficiency (KGE) values:", kge_values)