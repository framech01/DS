from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Set Korean font for matplotlib visualizations
set_korean_visualization()

# Function: train_prophet_model
# ----------------------------------------
# Trains a time-series forecasting model using Facebook Prophet to predict monthly accident rates.
#
# Parameters:
#     filepath (str): Path to the CSV file. The CSV must contain at least:
#                     - '기준월' (monthly date string)
#                     - '사고건수' (number of accidents per month)
#                     - '총_계' (total number of registered vehicles per month)
#
# Returns:
#     model (Prophet): A trained Prophet model fitted to the historical data.
#     df_prophet (pd.DataFrame): A processed DataFrame containing the original data used for training,
#                                 with the format: 'ds' (datetime), 'y' (accident rate).
def train_prophet_model(filepath):
    # Load and preprocess dataset
    df = load_and_prepare_data(filepath)
    df['YearMonth'] = pd.to_datetime(df['기준월']).dt.to_period('M')  # Extract year-month as period

    # Aggregate monthly accident and vehicle count
    monthly = df.groupby(['YearMonth']).agg({
        '사고건수': 'sum',
        '총_계': 'sum'
    }).reset_index()

    # Compute accident rate per 1,000 vehicles
    monthly['Accident Rate'] = (monthly['사고건수'] / (monthly['총_계'] + 1e-6)) * 1000
    monthly['YearMonth'] = monthly['YearMonth'].astype(str)  # Convert period to string for Prophet compatibility

    # Rename columns to fit Prophet input format: 'ds' for datetime, 'y' for target variable
    df_prophet = monthly[['YearMonth', 'Accident Rate']].rename(columns={'YearMonth': 'ds', 'Accident Rate': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])  # Ensure 'ds' is in datetime format

    # Instantiate Prophet model (additive model by default)
    # Prophet handles trend, seasonality, and holidays components.
    # Can be customized with add_seasonality(), add_country_holidays(), changepoint_prior_scale, etc.
    model = Prophet()
    model.fit(df_prophet)  # Fit model to the accident rate data

    # Generate forecast for 12 additional future months (default monthly frequency)
    future = model.make_future_dataframe(periods=12, freq='M')
    forecast = model.predict(future)  # Generate forecast DataFrame with columns: yhat, yhat_lower, yhat_upper, etc.

    # Evaluate model on historical data
    from sklearn.metrics import r2_score
    import numpy as np
    merged = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds')
    y_true = merged['y'].values  # Actual values
    y_pred = merged['yhat'].values  # Predicted values from Prophet
    mre = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))  # Mean Relative Error
    r2 = r2_score(y_true, y_pred)  # R-squared (explained variance)

    # Visualize full forecast including future and uncertainty intervals
    model.plot(forecast)
    plt.title('Monthly Accident Rate Forecast')
    plt.xlabel('Date')
    plt.ylabel('Accident Rate (per 1,000 vehicles)')
    plt.tight_layout()
    plt.show()

    return model, df_prophet
