import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd

from common import load_and_prepare_data, set_korean_visualization
from future_risk_forecasting import train_prophet_model
from vehicle_risk_clustering import train_vehicle_rf_model

# Configure visualization settings for proper Korean font rendering
set_korean_visualization()
filepath = "C:/Users/cjfwh/Downloads/최종_병합_교통사고_차량등록.csv"

# === Prophet Model Evaluation === #
print("\n[1] Prophet Accident Rate Forecasting Evaluation")

# Train the Prophet model using accident rate data\
model, df_prophet = train_prophet_model(filepath)

# Split dataset into training (80%) and testing (20%) sets
train_size = int(len(df_prophet) * 0.8)
train_df = df_prophet.iloc[:train_size]
test_df = df_prophet.iloc[train_size:]

# Create a dataframe for future predictions for the test period
future = model.make_future_dataframe(periods=len(test_df), freq='ME')
forecast = model.predict(future)

# Merge actual and predicted accident rates
merged = test_df.merge(forecast[['ds', 'yhat']], on='ds')

# Calculate evaluation metrics for Prophet
mae = mean_absolute_error(merged['y'], merged['yhat'])  # Mean Absolute Error
mre = ((merged['y'] - merged['yhat']).abs() / (merged['y'] + 1e-6)).mean()  # Mean Relative Error
r2 = r2_score(merged['y'], merged['yhat'])  # R-squared Score

# Output evaluation results
print(f"MAE: {mae:.4f}, MRE: {mre:.4f}, R^2: {r2:.4f}")

# Plot actual vs. predicted accident rates over time
plt.figure(figsize=(10, 6))
plt.plot(merged['ds'], merged['y'], label='Actual Accident Rate')
plt.plot(merged['ds'], merged['yhat'], label='Predicted Accident Rate', linestyle='--')
plt.title("Prophet Forecast vs Actual Accident Rate")
plt.xlabel("Month")
plt.ylabel("Accident Rate")
plt.legend()
plt.tight_layout()
plt.show()

# === RandomForest Model Evaluation === #
print("\n[2] Vehicle Composition-based Risk Classification (RandomForest)")

# Train the RandomForest classifier and get test data
rf_model, X_test, y_test, y_all = train_vehicle_rf_model(filepath)
y_pred = rf_model.predict(X_test)

# Display classification performance report
print("[RandomForest Classification Report]")
print(classification_report(y_test, y_pred))

# Generate confusion matrix and visualize it
cm = confusion_matrix(y_test, y_pred, labels=rf_model.classes_)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.title("Confusion Matrix (RandomForest)")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.tight_layout()
plt.show()

# === AdaBoost Cross-Validation Evaluation === #
print("\n[3] AdaBoost Cross-Validation Evaluation")

# Load and prepare the dataset
df = load_and_prepare_data(filepath)

# Aggregate average values by region and calculate vehicle composition ratios
grouped = df.groupby('시군구').agg({
    '승용_계': 'mean','승합_계': 'mean','화물_계': 'mean','특수_계': 'mean','총_계': 'mean','사고건수': 'mean'
}).reset_index()

# Add percentage features for each vehicle type
for col in ['승용', '승합', '화물', '특수']:
    grouped[f'{col}차 비율'] = grouped[f'{col}_계'] / (grouped['총_계'] + 1e-6)

# Select relevant features and normalize them
features = ['승용차 비율', '승합차 비율', '화물차 비율', '특수차 비율']
X = grouped[features]
X_scaled = StandardScaler().fit_transform(X)

# Assign risk labels based on accident count quantiles
thresholds = grouped['사고건수'].quantile([0.33, 0.66]).values
grouped['위험라벨'] = '중간위험 지역'
grouped.loc[grouped['사고건수'] <= thresholds[0], '위험라벨'] = '저위험 지역'
grouped.loc[grouped['사고건수'] >= thresholds[1], '위험라벨'] = '고위험 지역'
y = grouped['위험라벨']

# Perform 5-fold cross-validation with AdaBoost classifier
kf = KFold(n_splits=5, shuffle=True, random_state=42)
ada = AdaBoostClassifier(random_state=42)
scores = cross_val_score(ada, X_scaled, y, cv=kf, scoring='accuracy')

# Print mean and standard deviation of cross-validation accuracy
print("[AdaBoost Accuracy (5-Fold Mean ± Std)]", round(scores.mean(), 4), "+/-", round(scores.std(), 4))

# Visualize cross-validation accuracy
plt.figure(figsize=(5, 4))
plt.bar(['AdaBoost'], [scores.mean()], yerr=[scores.std()], color='lightcoral')
plt.title("AdaBoost Classification Accuracy (5-Fold)")
plt.ylim(0, 1)
plt.tight_layout()
plt.show()
