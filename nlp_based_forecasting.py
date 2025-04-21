import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
#pip install pandas matplotlib xgboost scikit-learn

# Load data
df = pd.read_csv('infra_data_with_text.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Target column
target = 'cpu_usage'

# TF-IDF on text column
vectorizer = TfidfVectorizer(max_features=10)  # You can increase features
text_features = vectorizer.fit_transform(df['event_desc'].fillna("")).toarray()
text_feature_names = vectorizer.get_feature_names_out()
tfidf_df = pd.DataFrame(text_features, columns=[f"text_{t}" for t in text_feature_names], index=df.index)

# Combine with numerical features
numerical_df = df[['memory_usage']].copy()
full_df = pd.concat([numerical_df, tfidf_df], axis=1)

# Lag features
for col in full_df.columns:
    full_df[f'{col}_lag_1'] = full_df[col].shift(1)
full_df.dropna(inplace=True)

# Labels
X = full_df
y = df[target].loc[full_df.index]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

# Train model
model = XGBRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"📉 RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_test.index, y_test, label="Actual")
plt.plot(y_test.index, y_pred, label="Predicted", linestyle='--')
plt.title("CPU Usage Forecast with Event Keywords")
plt.xlabel("Time")
plt.ylabel("CPU Usage (%)")
plt.legend()
plt.grid(True)
plt.show()
