import pandas as pd
import matplotlib.pyplot as plt

def apply_korean_font():
    """
    Apply Korean font settings for matplotlib visualizations to correctly display Korean characters.
    """
    import matplotlib.font_manager as fm
    font_path = 'C:/Windows/Fonts/malgun.ttf'
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
    plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is shown correctly

def load_data(filepath):
    """
    Load CSV data and extract the year from the '기준월' (base month) column.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with a new 'Year' column extracted from '기준월'.
    """
    df = pd.read_csv(filepath, encoding='utf-8')
    df['Year'] = pd.to_datetime(df['기준월']).dt.year
    return df
