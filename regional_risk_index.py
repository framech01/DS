from common import load_data, apply_korean_font
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

apply_korean_font()

def analyze_regional_risk(path):
    """
    Analyze accident and vehicle growth by region and compute composite risk scores.
    """
    df = load_data(path)

    # Aggregate yearly data per city
    summary = df.groupby(['시군구', 'Year']).agg({
        '사고건수': 'sum',
        '총_계': 'sum'
    }).reset_index()

    def growth_by_region(group, col):
        """
        Compute growth rate of a specific column grouped by city.
        """
        return group.groupby('시군구')[col].apply(
            lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-6) * 100
        )

    growth = pd.DataFrame({
        'Accident Growth': growth_by_region(summary, '사고건수'),
        'Vehicle Growth': growth_by_region(summary, '총_계')
    })

    growth['Basic Score'] = growth['Accident Growth'] / (growth['Vehicle Growth'] + 1e-6)

    # Compute severity per accident type
    df['Fatal'] = df['사망자수'] / (df['사고건수'] + 1e-6)
    df['Severe'] = df['중상자수'] / (df['사고건수'] + 1e-6)
    df['Minor'] = df['경상자수'] / (df['사고건수'] + 1e-6)

    severity = df.groupby('시군구').agg({
        'Fatal': 'mean',
        'Severe': 'mean',
        'Minor': 'mean'
    })

    # Combine risk scores
    risk_df = pd.concat([growth, severity], axis=1)
    risk_df['Composite Score'] = (
        risk_df['Basic Score'] +
        risk_df['Fatal'] * 100 +
        risk_df['Severe'] * 10 +
        risk_df['Minor']
    )

    # Merge city-province info
    info = df[['시군구', '시도']].drop_duplicates()
    risk_df = risk_df.merge(info, on='시군구', how='left')

    # Province-level summary
    province_df = risk_df.groupby('시도').agg({
        'Basic Score': 'mean',
        'Composite Score': 'mean'
    }).sort_values(by='Composite Score', ascending=False)

    # Heatmap 1
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.heatmap(province_df[['Basic Score']], cmap='Oranges', annot=True, fmt=".2f", ax=axes[0])
    axes[0].set_title("Basic Risk Score")
    sns.heatmap(province_df[['Composite Score']], cmap='Reds', annot=True, fmt=".2f", ax=axes[1])
    axes[1].set_title("Composite Risk Score")
    plt.tight_layout()
    plt.show()

    # Heatmap 2: risk ratio
    province_df['Risk Ratio'] = province_df['Basic Score'] / (province_df['Composite Score'] + 1e-6)
    plt.figure(figsize=(8, 6))
    sns.heatmap(province_df[['Risk Ratio']], cmap='PuBuGn', annot=True, fmt=".2f")
    plt.title("Risk Ratio")
    plt.tight_layout()
    plt.show()

    # Classify risk level based on ratio
    def classify_risk(ratio):
        if ratio <= -1.0:
            return 'Extreme'
        elif ratio <= -0.3:
            return 'High'
        elif ratio <= 0.3:
            return 'Moderate'
        else:
            return 'Low'

    province_df['Risk Level'] = province_df['Risk Ratio'].apply(classify_risk)

    # Final visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(y=province_df.index, x='Risk Ratio', hue='Risk Level', data=province_df, dodge=False, palette={
        'Extreme': '#d73027',
        'High': '#fc8d59',
        'Moderate': '#fee08b',
        'Low': '#91cf60'
    })
    plt.axvline(x=0, color='gray', linestyle='--')
    plt.title("Risk Level by Province")
    plt.xlabel("Risk Ratio")
    plt.ylabel("Province")
    plt.legend(title='Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
