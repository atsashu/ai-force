import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# 1. Load historical volume data
# Example: Replace with your real volume history
data = {
    'ds': pd.date_range(start='2022-01-01', periods=4, freq='YE'),
    'y': [
        70000, 75000, 70000,80000
    ]  # Volume data
}

data1 = pd.read_csv('D:/AI/myai-learn/castingdata.csv', parse_dates=['ds'])

# Step 2: Set date as index
#data1.set_index('ds', inplace=True)

# Step 3: Sort by date (very important for forecasting)
#data1.sort_index(inplace=True)
 

#df = pd.DataFrame(data1)

# 2. Forecast volume using Prophet
model = Prophet()
model.fit(data1)

# 3. Create future dates
future = model.make_future_dataframe(periods=3, freq='YE')

# 4. Predict future volume
forecast = model.predict(future)
#forecast.rename(columns={'yhat': 'predicted_capacity'}, inplace=True)
# 5. Derive capacity from forecasted volume (e.g., capacity = volume / 100)
forecast['capacity'] = forecast['yhat'] / 16800  # Change this rule as needed
#forecast['predicted_capacity'] = forecast['predicted_capacity'].clip(lower=0)
# 6. Plot results
plt.figure(figsize=(10, 6))
plt.plot(forecast['ds'], forecast['capacity'], label='Grid Capacity Required', color='purple')
plt.title('Forecasted Infrastructure Capacity (based on Volume)')
plt.xlabel('Date')
plt.ylabel('Capacity')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# 7. Optional: Save forecast to CSV
print(forecast[['ds',  'capacity']])
