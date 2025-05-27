from common import load_data, apply_korean_font
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

apply_korean_font()
# RandomForestClassifier for multi-class classification
# n_estimators: Number of trees in the forest (common range: 100–500)
# max_depth: Maximum depth of each decision tree (None = expand until all leaves are pure)
# min_samples_split: Minimum samples required to split a node (default: 2)
# min_samples_leaf: Minimum samples required at a leaf node (default: 1)
# max_features: Number of features to consider when looking for the best split (e.g., 'sqrt' for classification)
# random_state: Seed for reproducibility of results
# class_weight: Handle class imbalance (e.g., 'balanced' auto-adjusts weights inversely to class frequencies)
def cross_validate(X, y, k, trees):
    """
    Perform stratified k-fold validation using Random Forest and return average F1-score.
    """
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    accs, f1s, model = [], [], None

    for i, (tr_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        clf = RandomForestClassifier(n_estimators=trees, random_state=42)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)

        accs.append(accuracy_score(y_val, preds))
        f1s.append(f1_score(y_val, preds, average='macro'))

        print(f"Fold {i}: Accuracy = {accs[-1]:.4f}, F1 = {f1s[-1]:.4f}")
        if i == k:
            model = clf

    print("[Cross-validation Results]")
    print(f"Avg Accuracy: {np.mean(accs):.4f}")
    print(f"Avg F1 Score: {np.mean(f1s):.4f}")
    return np.mean(f1s), model

def train_vehicle_model(path):
    """
    Train and evaluate Random Forest classifier to cluster vehicle risk based on type composition.
    """
    df = load_data(path)
    print(f"Total entries after preprocessing: {df.shape[0]}")

    # Compute vehicle type ratios
    for col in ['승용', '승합', '화물', '특수']:
        df[f'{col}_Ratio'] = df[f'{col}_계'] / (df['총_계'] + 1e-6)

    features = ['승용_Ratio', '승합_Ratio', '화물_Ratio', '특수_Ratio']
    X = df[features]
    y_raw = df['사고건수']

    # Label risk levels based on quantiles
    thresholds = y_raw.quantile([0.33, 0.66]).values
    y = pd.cut(y_raw, bins=[-np.inf, thresholds[0], thresholds[1], np.inf],
               labels=['Low', 'Mid', 'High'])

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X), columns=features)
    df['Risk_Label'] = y.values

    # Pre-analysis visualization
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Risk_Label', y='사고건수', palette='Set2')
    plt.title('Accident Count by Risk Label')
    plt.tight_layout()
    plt.show()

    df.groupby('Risk_Label')[features].mean().plot(kind='bar', figsize=(10, 6), colormap='Set2')
    plt.title('Avg Vehicle Composition by Risk Label')
    plt.tight_layout()
    plt.show()

    # Run experiments over split ratios
    configs = [(0.6, 3, 100), (0.6, 10, 5), (0.7, 3, 100), (0.7, 10, 5), (0.8, 3, 100), (0.8, 10, 5)]
    best_f1, best_model, best_cfg = 0, None, None
    results = []

    for ratio, splits, trees in configs:
        X_tr, _, y_tr, _ = train_test_split(X_scaled, y, test_size=(1 - ratio), stratify=y, random_state=42)
        print(f"\n[Train {int(ratio*100)}% - splits={splits}, trees={trees}]")
        f1, model = cross_validate(X_tr, y_tr, splits, trees)
        results.append((ratio, splits, trees, f1))

        if f1 > best_f1:
            best_f1, best_model, best_cfg = f1, model, (ratio, splits, trees)

    # Print all results
    print("\n[All Configurations Summary]")
    for r, s, n, f in results:
        print(f"Train: {int(r*100)}%, Splits: {s}, Trees: {n} -> F1: {f:.4f}")

    print("\n[Best Configuration]")
    r, s, n = best_cfg
    print(f"Best -> Train: {int(r*100)}%, Splits: {s}, Trees: {n}, F1: {best_f1:.4f}")

    # Feature importance
    plt.figure(figsize=(8, 6))
    sns.barplot(x=best_model.feature_importances_, y=features, palette='Blues_d')
    plt.title('Feature Importance in Best RF Model')
    plt.tight_layout()
    plt.show()
