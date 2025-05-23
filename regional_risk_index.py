from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 설정
set_korean_visualization()

# 데이터 로딩 및 연도 추가
df = load_and_prepare_data("C:/Users/cjfwh/Downloads/최종_병합_교통사고_차량등록.csv")

# 집계
agg_df = df.groupby(['시군구', 'Year']).agg({
    '사고건수': 'sum',
    '총_계': 'sum'
}).reset_index()

# 증가율 계산 함수
def calc_growth(grouped, col):
    return grouped.groupby('시군구')[col].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-6) * 100
    )

# 기본 위험도 계산
growth = pd.DataFrame({
    'Accident Growth Rate': calc_growth(agg_df, '사고건수'),
    'Vehicle Growth Rate': calc_growth(agg_df, '총_계')
})
growth['Basic Risk Score'] = growth['Accident Growth Rate'] / (growth['Vehicle Growth Rate'] + 1e-6)

# 사고 심각도 계산
df['Fatality Rate'] = df['사망자수'] / (df['사고건수'] + 1e-6)
df['Severe Rate'] = df['중상자수'] / (df['사고건수'] + 1e-6)
df['Minor Rate'] = df['경상자수'] / (df['사고건수'] + 1e-6)

# 평균 치명도 계산
severity = df.groupby('시군구').agg({
    'Fatality Rate': 'mean',
    'Severe Rate': 'mean',
    'Minor Rate': 'mean'
})

# 종합 위험도 계산
region_df = pd.concat([growth, severity], axis=1)
region_df['Composite Risk Index'] = (
    region_df['Basic Risk Score'] + region_df['Fatality Rate'] * 100 + region_df['Severe Rate'] * 10 + region_df['Minor Rate']
)

# 시도 정보 추가
region_info = df[['시군구', '시도']].drop_duplicates()
region_df = region_df.merge(region_info, on='시군구', how='left')

# 시도별 평균 계산
sido_df = region_df.groupby('시도').agg({
    'Basic Risk Score': 'mean',
    'Composite Risk Index': 'mean'
}).sort_values(by='Composite Risk Index', ascending=False)

# 시각화 1
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(sido_df[['Basic Risk Score']], cmap='Oranges', annot=True, fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title("기본 위험도")
sns.heatmap(sido_df[['Composite Risk Index']], cmap='Reds', annot=True, fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title("종합 위험도")
plt.tight_layout()
plt.show()

# 비율 계산 및 시각화 2
sido_df['Risk Ratio'] = sido_df['Basic Risk Score'] / (sido_df['Composite Risk Index'] + 1e-6)
plt.figure(figsize=(8, 6))
sns.heatmap(sido_df[['Risk Ratio']], cmap='PuBuGn', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Risk Ratio")
plt.tight_layout()
plt.show()

# 등급화 함수
def classify_risk_ratio(r):
    if r <= -1.0:
        return '최고 위험'
    elif r <= -0.3:
        return '잠재적 고위험'
    elif r <= 0.3:
        return '중간 위험'
    else:
        return '낮은 위험'

# 등급 추가
sido_df['Risk Level'] = sido_df['Risk Ratio'].apply(classify_risk_ratio)

# 시각화 3
plt.figure(figsize=(10, 6))
sns.barplot(y=sido_df.index, x='Risk Ratio', hue='Risk Level', data=sido_df, dodge=False, palette={
    '최고 위험': '#d73027',
    '잠재적 고위험': '#fc8d59',
    '중간 위험': '#fee08b',
    '낮은 위험': '#91cf60'
})
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.title("Risk Ratio 및 등급")
plt.xlabel("Risk Ratio")
plt.ylabel("시도")
plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()