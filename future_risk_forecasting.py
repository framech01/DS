from common import load_data, apply_korean_font
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import os

# Apply Korean font settings for visualization
apply_korean_font()

def calc_metrics(actual, predicted):
    """
    Calculate regression evaluation metrics.
    """
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2

def run_prophet_analysis(path):
    """
    Train and evaluate Prophet model on accident rate forecasting with different data splits and seasonality.
    """
    os.makedirs("models", exist_ok=True)
    df = load_data(path)

    # Create month index and aggregate monthly accident data
    df['YearMonth'] = pd.to_datetime(df['기준월']).dt.to_period('M')
    monthly = df.groupby('YearMonth').agg({'사고건수': 'sum', '총_계': 'sum'}).reset_index()
    monthly['Rate'] = (monthly['사고건수'] / (monthly['총_계'] + 1e-6)) * 1000
    monthly['YearMonth'] = monthly['YearMonth'].astype(str)
    # Prophet model for time-series forecasting
    # daily_seasonality: Whether to include daily seasonality (True enables Fourier components for daily trends)
    # yearly_seasonality: Whether to include yearly seasonal patterns (True by default)
    # weekly_seasonality: Whether to include weekly seasonal effects (True by default)
    # changepoint_prior_scale: Flexibility of the trend change. Higher value allows the model to fit sudden changes better (e.g., 0.05 to 0.5)
    # seasonality_mode: 'additive' (default) or 'multiplicative'. Choose based on whether seasonal effects grow with the level of the series
    # holidays: DataFrame of holiday events to include as regressors (optional)
    # interval_width: Width of the uncertainty intervals (e.g., 0.95 = 95% confidence interval)
    # n_changepoints: Number of potential changepoints in the time series (default: 25)
    # Prepare for Prophet
    df_prophet = monthly.rename(columns={'YearMonth': 'ds', 'Rate': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

    split_ratios = [0.3, 0.8]
    results = []
    best = (None, None, float('inf'))

    for ratio in split_ratios:
        print(f"\n[Train Ratio {int(ratio*100)}% Evaluation]")
        total_len = len(df_prophet)
        train_len = int(total_len * ratio)

        train_df = df_prophet.iloc[:train_len]
        test_df = df_prophet.iloc[train_len:]

        for seasonal in [True, False]:
            label = 'with seasonality' if seasonal else 'no seasonality'
            print(f"\nProphet Model ({label})")

            model = Prophet(daily_seasonality=seasonal, changepoint_prior_scale=0.1)
            model.fit(train_df)

            future = model.make_future_dataframe(periods=len(test_df), freq='M')
            forecast = model.predict(future)

            y_true = test_df['y'].values
            y_pred = forecast.iloc[-len(test_df):]['yhat'].values

            mae, rmse, r2 = calc_metrics(y_true, y_pred)
            print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

            results.append((int(ratio * 100), label, mae, rmse, r2))
            if mae < best[2]:
                best = (int(ratio * 100), label, mae)

            # Visualization
            plt.figure(figsize=(10, 5))
            plt.plot(test_df['ds'], y_true, label='Actual')
            plt.plot(test_df['ds'], y_pred, label='Predicted')
            plt.title(f"Forecast vs Actual ({int(ratio*100)}% Train) - {label}")
            plt.xlabel("Date")
            plt.ylabel("Accident Rate")
            plt.legend()
            plt.tight_layout()
            plt.show()

    # Print summary of all configurations
    print("\n[Evaluation Summary]")
    for r, mode, m, s, r2 in results:
        print(f"Train: {r}%, Mode: {mode} -> MAE: {m:.4f}, RMSE: {s:.4f}, R²: {r2:.4f}")

    print("\n[Best Configuration (Lowest MAE)]")
    print(f"Best -> Train: {best[0]}%, Mode: {best[1]}, MAE: {best[2]:.4f}")
