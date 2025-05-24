from common import load_and_prepare_data, set_korean_visualization
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set Korean font for consistent visualization
set_korean_visualization()

# Function: train_vehicle_rf_model
# -------------------------------------------
# Trains a Random Forest classifier to predict regional accident risk levels based on vehicle composition.
#
# Parameters:
#     filepath (str): Path to the CSV file containing vehicle counts and accident data per region.
#                     Required columns: '시군구', '사고건수', '승용_계', '승합_계', '화물_계', '특수_계', '총_계'.
#
# Returns:
#     clf (RandomForestClassifier): Trained Random Forest model.
#     X_test (np.ndarray): Feature set used for testing.
#     y_test (pd.Series): Actual risk labels for the test set.
#     y (pd.Series): Full set of risk labels for the dataset.
def train_vehicle_rf_model(filepath):
    # Load dataset and compute yearly average per region
    df = load_and_prepare_data(filepath)
    grouped = df.groupby('시군구').agg({
        '승용_계': 'mean',
        '승합_계': 'mean',
        '화물_계': 'mean',
        '특수_계': 'mean',
        '총_계': 'mean',
        '사고건수': 'mean'
    }).reset_index()

    # Calculate ratios for each vehicle type
    for col in ['승용', '승합', '화물', '특수']:
        grouped[f'{col}차 비율'] = grouped[f'{col}_계'] / (grouped['총_계'] + 1e-6)

    # Define features and normalize them
    features = ['승용차 비율', '승합차 비율', '화물차 비율', '특수차 비율']
    X = grouped[features]
    X_scaled = StandardScaler().fit_transform(X)

    # Define risk labels based on accident quantiles (low/mid/high)
    thresholds = grouped['사고건수'].quantile([0.33, 0.66]).values
    grouped['위험라벨'] = '중간위험 지역'
    grouped.loc[grouped['사고건수'] <= thresholds[0], '위험라벨'] = '저위험 지역'
    grouped.loc[grouped['사고건수'] >= thresholds[1], '위험라벨'] = '고위험 지역'
    y = grouped['위험라벨']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Instantiate and train Random Forest classifier
    # RandomForestClassifier is a robust ensemble model using multiple decision trees
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    # --- Visualization Part ---
    # Plot boxplot of accident count by risk level
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=grouped, x='위험라벨', y='사고건수', palette='Set2')
    plt.title('Distribution of Accident Counts by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Average Accident Count')
    plt.tight_layout()
    plt.show()

    # Plot vehicle type ratio per risk level group
    risk_summary = grouped.groupby('위험라벨')[features].mean()
    risk_summary.plot(kind='bar', figsize=(10, 6), colormap='Set2')
    plt.title('Average Vehicle Type Composition by Risk Level')
    plt.xlabel('Risk Level')
    plt.ylabel('Vehicle Type Ratio')
    plt.legend(title='Vehicle Type')
    plt.tight_layout()
    plt.show()

    # Plot feature importances from the Random Forest model
    importances = clf.feature_importances_
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances, y=features, palette='Blues_d')
    plt.title('Feature Importance in Random Forest Model')
    plt.tight_layout()
    plt.show()

    return clf, X_test, y_test, y
