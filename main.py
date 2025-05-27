from regional_risk_index import analyze_regional_risk
from vehicle_risk_clustering import train_vehicle_model
from future_risk_forecasting import run_prophet_analysis

DATA_PATH = "C:/Users/cjfwh/Downloads/최종_병합_교통사고_차량등록.csv"

print("[1] Regional Risk Index Analysis")
analyze_regional_risk(DATA_PATH)

print("[2] Training RandomForest model for Vehicle Risk Clustering")
train_vehicle_model(DATA_PATH)

print("[3] Evaluating RandomForest model")
print("[4] Training Prophet model for Future Risk Forecasting")
run_prophet_analysis(DATA_PATH)
