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


# Function to create sequences for LSTM
def create_sequences(data, n_past, n_future):
    X, y = [], []
    for i in range(len(data) - n_past - n_future + 1):
        X.append(data[i:i + n_past])
        y.append(data[i + n_past:i + n_past + n_future])
    return np.array(X), np.array(y)







# Initialize an empty dictionary to store performance metrics
r2_values = {}
mse_values = {}
kge_values = {}

# Loop through each feature
for feature in features:
    # Loop through each lag
    for lag in lags:

        # Define the target variable
        target_column  = 'streamflow'
        
        X_seq, y_seq = create_sequences(df[[f'{feature}_{lag}_lag' for feature in features for lag in lags] + [target_column]],
                                 n_past=14, n_future=1)
        
        # Train-test split
        split_point = int(0.8 * len(X_seq))
        X_train, X_test = X_seq[:split_point], X_seq[split_point:]
        y_train, y_test = y_seq[:split_point], y_seq[split_point:]
        
        # Normalize the data
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        
        X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
        X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
        
        y_train = scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
        y_test = scaler_y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)
        # Build and train the LSTM model
        model = Sequential()
        model.add(LSTM(units=50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)
        
        # Evaluate the model
        loss = model.evaluate(X_test, y_test)
        print(f'Test Loss: {loss}')
        
        # Make predictions
        y_pred = model.predict(X_test)

        

# =============================================================================
#         
#         n = len(X)
#         aa = 0.7 * n
#         aa = int(aa)
#         bb = 0.15 * n
#         bb = int(bb)+aa
# 
# 
#        
# =============================================================================
# =============================================================================
#         X_train, X_valid, X_test = X[:aa], X[aa:bb], X[bb:]
#         Y_train, Y_valid, Y_test = y[:aa], y[aa:bb], y[bb:]
# 
#         X_train, Y_train = shuffle(X_train, Y_train, random_state=42)
#         
# 
#         # Normalize the data
#         scaler_X = StandardScaler()
#         scaler_y = StandardScaler()
# 
#         X_train = scaler_X.fit_transform(X_train)
#         X_test = scaler_X.transform(X_test)
#         X_valid = scaler_X.transform(X_valid)
# 
#         Y_train = scaler_y.fit_transform(Y_train.values.reshape(-1, 1))
# 
#         Y_test = scaler_y.transform(Y_test.values.reshape(-1, 1))
#         Y_valid = scaler_y.transform(Y_valid.values.reshape(-1, 1))
# 
# 
#         n_past = 1
# 
#         x_train = []
#         y_train = []
#         x_valid = []
#         y_valid = []
#         x_test = []
#         y_test = []
#         print(Y_train)
#         
#         
#         for i in range(0, len(X_train) - n_past + 1):
#             x_train.append(X_train[i: i + n_past])
# 
#             y_train.append(Y_train[i + n_past -1])
#         print(y_train)
#         aaa
#         
#         for i in range(0, len(X_valid) - n_past + 1):
#             x_valid.append(X_valid[i: i + n_past])
#             y_valid.append(Y_valid[i + n_past])
#         
#         
#         for i in range(0, len(X_test) - n_past + 1):
#             x_test.append(X_test[i: i + n_past])
#             y_test.append(Y_test[i + n_past])
# 
# 
#         # Reshape input data for LSTM
#         x_train = x_train.reshape((x_train.shape[0], 1, x_train.shape[1]))
#         x_test = x_test.reshape((x_test.shape[0], 1, x_test.shape[1]))
#         x_valid = x_valid.reshape((x_valid.shape[0], 1, X_test.shape[1]))
# 
# =============================================================================

# =============================================================================
# 
#         X_train = []
#         Y_train = []
#         
#         # Include the last "lag" days before the current day for each training sample
#         for i in range(lag, aa):
#             X_train.append(X.iloc[i-lag:i, :].values.flatten())
#             Y_train.append(y.iloc[i])
#         
#         X_train = np.array(X_train)
#         Y_train = np.array(Y_train)
#         
#         print(len(X_train))
#         aaaa
# 
# 
# 
#         # Build LSTM model
#         model = Sequential()
#         model.add(LSTM(units=50, input_shape=(x_train.shape[1], x_train.shape[2])))
#         model.add(Dense(units=1, activation='linear'))
#         model.compile(optimizer='adam', loss='mean_squared_error')
# 
#         # Train the model
#         model.fit(x_train, y_train, epochs=50, batch_size=32, validation_data=(x_test, y_test))
# 
#         # Predictions
#         y_pred = model.predict(x_test)
# =============================================================================

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
