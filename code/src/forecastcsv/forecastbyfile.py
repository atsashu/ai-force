import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

# Load data
df = pd.read_csv('infraparams.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=False)

# Create output folder
os.makedirs("forecasts", exist_ok=True)

# Iterate over each metric column
for column in df.columns:
    if column == 'timestamp':
        continue

    print(f"\nForecasting for: {column}")

    # Prepare data for Prophet
    data = df[['timestamp', column]].dropna().rename(columns={
        'timestamp': 'ds',
        column: 'y'
    })

    # Initialize and fit model
    model = Prophet()
    model.fit(data)

    # Future 30 days
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # Save to CSV
    output_file = f"forecasts/{column}_forecast.csv"
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(output_file, index=False)

    # Plot
    fig = model.plot(forecast)
    plt.title(f"Forecast for {column}")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.tight_layout()
    plt.savefig(f"forecasts/{column}_forecast.png")
    plt.close()

print("\n✅ All forecasts generated in the 'forecasts' folder.")
