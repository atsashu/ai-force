import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import re

# Load data
df = pd.read_csv('code/src/forecastcsv/infraparams.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=False)

# Create output folder
output_folder = "forecasts"
os.makedirs(output_folder, exist_ok=True)
print(f"Output folder exists: {os.path.exists(output_folder)}")

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
    model = Prophet(yearly_seasonality=True)  # Enabling yearly seasonality explicitly
    model.fit(data)

    # Forecast for next 5 year-end points
    future = model.make_future_dataframe(periods=5, freq='YE')
    forecast = model.predict(future)

    # Prepare and round output
    output_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted_Capacity',
        'yhat_lower': 'Lower_Bound',
        'yhat_upper': 'Upper_Bound'
    })

    # Round numeric columns to integers
    output_df[['Predicted_Capacity', 'Lower_Bound', 'Upper_Bound']] = (
        output_df[['Predicted_Capacity', 'Lower_Bound', 'Upper_Bound']].round(0).astype(int)
    )

    # Sanitize column name for file paths (replace special characters)
    safe_column = re.sub(r'[^\w\-_\. ]', '_', column)

    # Check if output_df is empty before saving
    if output_df.empty:
        print(f"No forecast data for {column}")
    else:
        output_file = f"{output_folder}/{safe_column}_forecast.csv"
        output_df.to_csv(output_file, index=False)
        print(f"Saved forecast for {column} at {output_file}")

    # Save plot as PNG
    plot_file = f"{output_folder}/{safe_column}_forecast.png"
    plt.figure()
    fig = model.plot(forecast)
    plt.title(f"Forecast for {column}")
    plt.xlabel("Date")
    plt.ylabel(column)
    plt.tight_layout()
    
    # Save plot and check if it exists
    plt.savefig(plot_file)
    if os.path.exists(plot_file):
        print(f"Plot saved for {column} at {plot_file}")
    else:
        print(f"Failed to save plot for {column}")

print("\n✅ All forecasts generated in the 'forecasts' folder.")
