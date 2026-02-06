import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pmdarima as pm
from sklearn.metrics import mean_squared_error

# ==========================================
# 1. Generate Synthetic Data
# ==========================================
np.random.seed(42)

# Create a time index (10 years of monthly data)
dates = pd.date_range(start='2010-01-01', periods=120, freq='MS')

# Components: Linear Trend + Seasonality + Random Noise
values = np.linspace(10, 50, len(dates))  # Linear Trend
values += 10 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)  # Yearly Seasonality
values += np.random.normal(0, 2, len(dates))  # Random Noise

# Create Pandas Series
data = pd.Series(values, index=dates)

# Split into Train (80%) and Test (20%)
train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]

print(f"Data Generated: {len(data)} total points.")
print(f"Training samples: {len(train)}, Testing samples: {len(test)}")
print("-" * 30)

# ==========================================
# 2. Automated Model Selection (Auto-ARIMA)
# ==========================================
print("Running Auto-ARIMA to find best parameters...")

# auto_arima will search for the optimal (p,d,q) and (P,D,Q)
model = pm.auto_arima(
    train,
    seasonal=True,      # Account for seasonality
    m=12,               # Frequency of the cycle (12 months)
    d=None,             # Let model determine 'd' (differencing)
    test='kpss',        # Statistical test for stationarity
    trace=True,         # Print the search progress
    error_action='ignore',   
    suppress_warnings=True, 
    stepwise=True       # Smart search (faster than grid search)
)

print("-" * 30)
print("Best Model Found:")
print(model.summary())

# ==========================================
# 3. Forecasting & Evaluation
# ==========================================
# Predict for the length of the test set
prediction, conf_int = model.predict(n_periods=len(test), return_conf_int=True)

# Convert prediction to pandas Series for easier plotting
prediction_series = pd.Series(prediction, index=test.index)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(test, prediction))
print(f"\nModel RMSE: {rmse:.4f}")

# ==========================================
# 4. Visualization
# ==========================================
plt.figure(figsize=(14, 7))

# Plot historical training data
plt.plot(train.index, train, label='Training Data', color='gray', alpha=0.6)

# Plot actual test data
plt.plot(test.index, test, label='Actual Test Data', color='blue', linewidth=2)

# Plot the Forecast
plt.plot(prediction_series.index, prediction_series, label=f'Auto-ARIMA Forecast (RMSE={rmse:.2f})', color='red', linestyle='--')

# Plot Confidence Intervals (shaded area)
plt.fill_between(
    test.index,
    conf_int[:, 0], # Lower bound
    conf_int[:, 1], # Upper bound
    color='red', 
    alpha=0.1, 
    label='Confidence Interval (95%)'
)

plt.title(f'Auto-ARIMA Forecast: {model.order} x {model.seasonal_order}')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend(loc='upper left')
plt.grid(True)
plt.savefig('arima_forecast.png')
print("Plot saved to arima_forecast.png")
plt.close()