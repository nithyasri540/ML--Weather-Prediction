# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset



## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset and separate it into input features (e.g., humidity, wind speed) and target variables (temperature, PM2.5, energy).
2.Generate several decision trees using random subsets of the data and features (bootstrap sampling). 
3. Each tree makes its own prediction; combine all predictions (average for regression problems).
4. Compare predicted values with actual values using metrics like Mean Squared Error (MSE) to check accuracy

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: S.NITHYASRI
RegisterNumber: 25018590
import pandas as pd
import numpy as np
data = pd.read_csv("C:/Users/acer/Downloads/weather-station-eee-block_2024_07_13.csv")
print(data.head())
print(data.isnull().sum())
data.fillna(method='ffill', inplace=True)
X = data[['hum', 'pressure', 'wind_speed']]
y_temp = data['tem']
y_pm = data['pm2_5']
y_energy = data['tsr']
from sklearn.model_selection import train_test_split

X_train, X_test, y_temp_train, y_temp_test = train_test_split(X, y_temp, test_size=0.2, random_state=0)
X_train, X_test, y_pm_train, y_pm_test = train_test_split(X, y_pm, test_size=0.2, random_state=0)
X_train, X_test, y_energy_train, y_energy_test = train_test_split(X, y_energy, test_size=0.2, random_state=0)
from sklearn.ensemble import RandomForestRegressor

model_temp = RandomForestRegressor(n_estimators=100, random_state=0)
model_pm = RandomForestRegressor(n_estimators=100, random_state=0)
model_energy = RandomForestRegressor(n_estimators=100, random_state=0)
model_temp.fit(X_train, y_temp_train)
model_pm.fit(X_train, y_pm_train)
model_energy.fit(X_train, y_energy_train)
temp_pred = model_temp.predict(X_test)
pm_pred = model_pm.predict(X_test)
energy_pred = model_energy.predict(X_test)
from sklearn.metrics import mean_squared_error

print("Temperature MSE:", mean_squared_error(y_temp_test, temp_pred))
print("PM2.5 MSE:", mean_squared_error(y_pm_test, pm_pred))
print("Energy MSE:", mean_squared_error(y_energy_test, energy_pred))
 
*/
```

## Output:
<img width="340" height="65" alt="Screenshot 2026-03-17 132814" src="https://github.com/user-attachments/assets/acc02712-6e16-4099-9793-fef4f28b0c51" />




## Result:
