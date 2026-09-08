import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def apply_korean_font():
    """
    Apply Korean font settings for matplotlib visualizations to correctly display Korean characters.
    """
    import matplotlib.font_manager as fm
    candidates = [Path("C:/Windows/Fonts/malgun.ttf"), Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf")]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path:
        plt.rc("font", family=fm.FontProperties(fname=font_path).get_name())
    plt.rcParams['axes.unicode_minus'] = False  # Ensure minus sign is shown correctly

def load_data(filepath):
    """
    Load CSV data and extract the year from the '기준월' (base month) column.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with a new 'Year' column extracted from '기준월'.
    """
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    df = pd.read_csv(path, encoding='utf-8', low_memory=False)
    required = {'기준월', '시도', '시군구', '사고건수', '총_계'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"필수 컬럼이 없습니다: {sorted(missing)}")
    df['기준월'] = pd.to_datetime(df['기준월'], errors='raise')
    df['Year'] = df['기준월'].dt.year
    return df


# Backward-compatible helpers used by the standalone EDA scripts.
load_and_prepare_data = load_data
set_korean_visualization = apply_korean_font
