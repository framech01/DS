from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 기본 설정
sns.set(style="whitegrid", font="Malgun Gothic", rc={"axes.unicode_minus": False})
plt.rcParams["figure.figsize"] = (12, 6)

# 데이터 로딩
df = load_and_prepare_data("C:/Users/a0104/Downloads/최종_병합_교통사고_차량등록.csv")

# 기준월을 datetime으로 변환하고 연도 추출
df['기준월'] = pd.to_datetime(df['기준월'])
df['연도'] = df['기준월'].dt.year

# ----------------------------------------
# 1. 시도별 연간 사고건수 비교
# ----------------------------------------
accident_by_year_province = df.groupby(['시도', '연도'])['사고건수'].sum().reset_index()
pivot_acc_province = accident_by_year_province.pivot(index='시도', columns='연도', values='사고건수').fillna(0)

pivot_acc_province.plot(kind='bar', figsize=(12, 6), colormap='Set2')
plt.title('시도별 연간 사고건수 비교')
plt.ylabel('사고건수')
plt.xlabel('시도')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------
# 2. 시도별 연간 차량 등록 대수 비교
# ----------------------------------------
car_by_year_province = df.groupby(['시도', '연도'])['총_계'].sum().reset_index()
pivot_car = car_by_year_province.pivot(index='시도', columns='연도', values='총_계').fillna(0)

pivot_car.plot(kind='bar', figsize=(12, 6), colormap='Set1')
plt.title('시도별 연간 차량 등록 대수 비교')
plt.ylabel('총 차량 수')
plt.xlabel('시도')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# ----------------------------------------
# 3. 시도별 연도별 총 차량수 vs 사고건수 산점도
# (연도 단위, 시도별 평균)
# ----------------------------------------
grouped_scatter = df.groupby(['연도', '시도'])[['사고건수', '총_계']].mean().reset_index()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=grouped_scatter, x='총_계', y='사고건수', hue='연도', style='시도', s=100)
plt.title('시도별 연도별 총 차량수 vs 사고건수 (시도 평균)')
plt.xlabel('총 차량 수 (시도 평균)')
plt.ylabel('사고건수 (시도 평균)')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()



# 4. 상관관계 분석 (교통지표 중심)
corr_cols = ['총_계', '사고건수', '사망자수', '부상자수', '경상자수', '중상자수']
corr_matrix = df[corr_cols].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('교통사고 관련 변수 간 상관관계')
plt.show()
