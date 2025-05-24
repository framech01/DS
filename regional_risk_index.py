from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Enable Korean font rendering in plots
set_korean_visualization()

# Load dataset and extract year from month field
df = load_and_prepare_data("C:/Users/cjfwh/Downloads/최종_병합_교통사고_차량등록.csv")

# Aggregate total accidents and vehicle counts per city and year
agg_df = df.groupby(['시군구', 'Year']).agg({
    '사고건수': 'sum',
    '총_계': 'sum'
}).reset_index()

# Helper function to compute growth rate between first and last year per city
def calc_growth(grouped, col):
    return grouped.groupby('시군구')[col].apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / (x.iloc[0] + 1e-6) * 100
    )

# Compute growth rates for accidents and vehicle registration
growth = pd.DataFrame({
    'Accident Growth Rate': calc_growth(agg_df, '사고건수'),
    'Vehicle Growth Rate': calc_growth(agg_df, '총_계')
})

# Calculate base risk score: accident growth normalized by vehicle growth
growth['Basic Risk Score'] = growth['Accident Growth Rate'] / (growth['Vehicle Growth Rate'] + 1e-6)

# Calculate severity indicators: death, serious injury, and minor injury per accident
df['Fatality Rate'] = df['사망자수'] / (df['사고건수'] + 1e-6)
df['Severe Rate'] = df['중상자수'] / (df['사고건수'] + 1e-6)
df['Minor Rate'] = df['경상자수'] / (df['사고건수'] + 1e-6)

# Compute average severity per city
severity = df.groupby('시군구').agg({
    'Fatality Rate': 'mean',
    'Severe Rate': 'mean',
    'Minor Rate': 'mean'
})

# Combine all risk factors into a regional risk index
region_df = pd.concat([growth, severity], axis=1)
region_df['Composite Risk Index'] = (
    region_df['Basic Risk Score'] +
    region_df['Fatality Rate'] * 100 +
    region_df['Severe Rate'] * 10 +
    region_df['Minor Rate']
)

# Append city-province mapping for grouping
region_info = df[['시군구', '시도']].drop_duplicates()
region_df = region_df.merge(region_info, on='시군구', how='left')

# Calculate average risk per province (시도)
sido_df = region_df.groupby('시도').agg({
    'Basic Risk Score': 'mean',
    'Composite Risk Index': 'mean'
}).sort_values(by='Composite Risk Index', ascending=False)

# Visualization 1: Heatmaps of basic and composite risk scores
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(sido_df[['Basic Risk Score']], cmap='Oranges', annot=True, fmt=".2f", linewidths=0.5, ax=axes[0])
axes[0].set_title("Basic Risk Score")
sns.heatmap(sido_df[['Composite Risk Index']], cmap='Reds', annot=True, fmt=".2f", linewidths=0.5, ax=axes[1])
axes[1].set_title("Composite Risk Index")
plt.tight_layout()
plt.show()

# Visualization 2: Risk ratio = basic / composite
sido_df['Risk Ratio'] = sido_df['Basic Risk Score'] / (sido_df['Composite Risk Index'] + 1e-6)
plt.figure(figsize=(8, 6))
sns.heatmap(sido_df[['Risk Ratio']], cmap='PuBuGn', annot=True, fmt=".2f", linewidths=0.5)
plt.title("Risk Ratio")
plt.tight_layout()
plt.show()

# Risk level classification rule based on ratio thresholds
def classify_risk_ratio(r):
    if r <= -1.0:
        return 'Extreme Risk'
    elif r <= -0.3:
        return 'Potential High Risk'
    elif r <= 0.3:
        return 'Moderate Risk'
    else:
        return 'Low Risk'

# Apply classification and visualize
sido_df['Risk Level'] = sido_df['Risk Ratio'].apply(classify_risk_ratio)
plt.figure(figsize=(10, 6))
sns.barplot(y=sido_df.index, x='Risk Ratio', hue='Risk Level', data=sido_df, dodge=False, palette={
    'Extreme Risk': '#d73027',
    'Potential High Risk': '#fc8d59',
    'Moderate Risk': '#fee08b',
    'Low Risk': '#91cf60'
})
plt.axvline(x=0, color='gray', linestyle='--', linewidth=1)
plt.title("Risk Ratio and Levels by Province")
plt.xlabel("Risk Ratio")
plt.ylabel("Province")
plt.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()
