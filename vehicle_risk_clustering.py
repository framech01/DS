from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# 설정
set_korean_visualization()

# 데이터 로딩
df = load_and_prepare_data("C:/Users/cjfwh/Downloads/최종_병합_교통사고_차량등록.csv")

# 평균 계산
grouped = df.groupby('시군구').agg({
    '승용_계': 'mean',
    '승합_계': 'mean',
    '화물_계': 'mean',
    '특수_계': 'mean',
    '총_계': 'mean',
    '사고건수': 'mean'
}).reset_index()

# 차량 비율 계산
for col in ['승용', '승합', '화물', '특수']:
    grouped[f'{col}차 비율'] = grouped[f'{col}_계'] / (grouped['총_계'] + 1e-6)

# 클러스터링
features = ['승용차 비율', '승합차 비율', '화물차 비율', '특수차 비율']
X = grouped[features]
X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, random_state=42)
grouped['클러스터'] = kmeans.fit_predict(X_scaled)

# 위험라벨 지정
summary = grouped.groupby('클러스터')['사고건수'].mean()
high_risk = summary.idxmax()
low_risk = summary.idxmin()
grouped['위험라벨'] = '중간위험 지역'
grouped.loc[grouped['클러스터'] == high_risk, '위험라벨'] = '고위험 지역'
grouped.loc[grouped['클러스터'] == low_risk, '위험라벨'] = '저위험 지역'

# 시각화 1
plt.figure(figsize=(8, 6))
sns.boxplot(data=grouped, x='위험라벨', y='사고건수', palette='Set2')
plt.title('사고건수 분포')
plt.xlabel('위험 유형')
plt.ylabel('사고건수')
plt.tight_layout()
plt.show()

# 시각화 2
risk_summary = grouped.groupby('위험라벨')[features].mean()
risk_summary.plot(kind='bar', figsize=(10, 6), colormap='Set2')
plt.title('차량 비율')
plt.xlabel('위험 유형')
plt.ylabel('차량 비율')
plt.legend(title='차량 종류')
plt.tight_layout()
plt.show()

# 예측 모델 학습
y = grouped['위험라벨']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# 변수 중요도 시각화
plt.figure(figsize=(8, 6))
sns.barplot(x=clf.feature_importances_, y=features, palette='Blues_d')
plt.title('변수 중요도')
plt.tight_layout()
plt.show()

# 샘플 예측
sample_input = pd.DataFrame({
    '승용차 비율': [0.84],
    '승합차 비율': [0.03],
    '화물차 비율': [0.12],
    '특수차 비율': [0.01]
})
predicted_risk = clf.predict(sample_input)
print("예측된 위험유형:", predicted_risk[0])
