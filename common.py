import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure matplotlib to support Korean font and display minus signs correctly.
def set_korean_visualization():
    plt.rcParams['font.family'] = 'Malgun Gothic'  # Set font for Korean characters
    plt.rcParams['axes.unicode_minus'] = False     # Ensure minus signs are rendered correctly

# Load CSV data and extract year from the date column.
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')  # Load data from CSV with UTF-8 encoding
    df['Year'] = pd.to_datetime(df['기준월']).dt.year  # Extract year from '기준월' column
    return df
