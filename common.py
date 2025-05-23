import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 시각화 설정
def set_korean_visualization():
    plt.rcParams['font.family'] = 'Malgun Gothic'
    plt.rcParams['axes.unicode_minus'] = False

# 데이터 로딩 및 연도 추가
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    df['Year'] = pd.to_datetime(df['기준월']).dt.year
    return df