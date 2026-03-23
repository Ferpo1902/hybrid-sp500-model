import pandas as pd
import numpy as np
import lightgbm as lgb
import yfinance as yf
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define the backtest period
start_date = '2016-01-01'
end_date = '2026-01-01'

# Fetch SPY data
spy_data = yf.download('SPY', start=start_date, end=end_date)

# Implement Buy & Hold Strategy
spy_data['Buy_Hold'] = (spy_data['Close'] / spy_data['Close'].iloc[0]) - 1

# Function for Modelo Anterior
def modelo_anterior(X_train, y_train, X_test):
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Function for Modelo Nuevo
def modelo_nuevo(X_train, y_train, X_test):
    model = lgb.LGBMRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Prepare your macro features and split data here
# For demonstration, let's assume we have features and targets ready
# X_train, y_train, X_test, y_test = ...

# Backtest logic for models
spy_data['Modelo_Anterior'] = modelo_anterior(X_train, y_train, x_test)
spy_data['Modelo_Nuevo'] = modelo_nuevo(X_train, y_train, x_test)

# Calculate metrics
spy_data['Modelo_Anterior_Returns'] = (spy_data['Modelo_Anterior'] - spy_data['Modelo_Anterior'].shift(1)) / spy_data['Modelo_Anterior'].shift(1)
spy_data['Modelo_Nuevo_Returns'] = (spy_data['Modelo_Nuevo'] - spy_data['Modelo_Nuevo'].shift(1)) / spy_data['Modelo_Nuevo'].shift(1)

# Visualizing results
plt.figure(figsize=(14, 7))
plt.plot(spy_data['Buy_Hold'], label='SPY Buy & Hold', color='blue')
plt.plot(spy_data['Modelo_Anterior'], label='Modelo Anterior', color='orange')
plt.plot(spy_data['Modelo_Nuevo'], label='Modelo Nuevo', color='green')
plt.title('Backtest Comparison 2016-2026')
plt.xlabel('Date')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()