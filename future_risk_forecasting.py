from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import r2_score
import numpy as np

# 설정
set_korean_visualization()

# 데이터 로딩 및 연도월 추가
df = load_and_prepare_data("C:/Users/cjfwh/Downloads/최종_병합_교통사고_차량등록.csv")
df['YearMonth'] = pd.to_datetime(df['기준월']).dt.to_period('M')

# 사고율 계산
monthly = df.groupby(['YearMonth']).agg({
    '사고건수': 'sum',
    '총_계': 'sum'
}).reset_index()
monthly['Accident Rate'] = (monthly['사고건수'] / (monthly['총_계'] + 1e-6)) * 1000
monthly['YearMonth'] = monthly['YearMonth'].astype(str)

# Prophet 입력 준비
df_prophet = monthly[['YearMonth', 'Accident Rate']].rename(columns={'YearMonth': 'ds', 'Accident Rate': 'y'})
df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])

# 예측 모델 학습
model = Prophet()
model.fit(df_prophet)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)

# 정확도 계산
merged = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds')
y_true = merged['y'].values
y_pred = merged['yhat'].values
mre = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6)))
r2 = r2_score(y_true, y_pred)
print("Mean Relative Error (MRE):", round(mre, 4))
print("R^2 Score:", round(r2, 4))

# 시각화
model.plot(forecast)
plt.title('월별 사고율 예측')
plt.xlabel('기준월')
plt.ylabel('사고율')
plt.tight_layout()
plt.show()
