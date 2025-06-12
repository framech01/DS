from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 기본 설정
sns.set(style="whitegrid", font="Malgun Gothic", rc={"axes.unicode_minus": False})
plt.rcParams["figure.figsize"] = (12, 6)

# 데이터 로딩
df = load_and_prepare_data("C:/Users/a0104/Downloads/최종_병합_교통사고_차량등록.csv")
df['연도'] = df['기준월'].str.split('-').str[0]
num_cols = ['총_계', '사고건수', '사망자수', '부상자수', '경상자수', '중상자수']


# 1. 결측치 확인
print("🔍 결측치 확인:")
print(df.isnull().sum())

# 2. 데이터 타입 확인
print("\n📘 데이터 타입:")
print(df.dtypes)

# 3. 기초 통계 확인
print("\n📊 기초 통계 요약:")
print(df.describe())

# 4. 중복 데이터 확인
print("\n📋 중복 행 수:")
print(df.duplicated().sum())

# 5. 시도, 시군구, 연도 고유값
print("\n📍 시도 목록:", df['시도'].unique())
print("📍 시군구 수:", df['시군구'].nunique())
print("📍 연도별 데이터 수:\n", df['연도'].value_counts())

# 6. 주요 수치형 변수 히스토그램

for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], bins=20, kde=True)
    plt.title(f'📊 {col} 분포')
    plt.xlabel(col)
    plt.ylabel('빈도')
    plt.tight_layout()
    plt.show()

# 7. 이상치 시각화 (Boxplot)
for col in num_cols:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df[[col]])
    plt.title(f'📦 {col} 이상치 확인 (Boxplot)')
    plt.tight_layout()
    plt.show()
    #시군구 지역별로 편차 심해서 이상치가 많이 뛰는듯?
