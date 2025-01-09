import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

data_path = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

numerical_features = ["BMI", "Age", "PhysHlth", "MentHlth", "GenHlth", "Education", "Income"]
categorical_features = [col for col in X.columns if col not in numerical_features]

numerical_features2 = ["Age", "GenHlth", "Education", "Income", "PhysHlth", "MentHlth"]
bmi = ["BMI"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("bmi", StandardScaler(), bmi),
        ("num", MinMaxScaler(), numerical_features2),
        ("cat", OneHotEncoder(drop='first'), categorical_features)
    ]
)
X_preprocessed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)

models = {
    "Logistic Regression": LogisticRegression(max_iter=500),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
    "LightGBM": LGBMClassifier(),
    "CatBoost": CatBoostClassifier()
}

model_colors = {
    "Logistic Regression": "#1f77b4",
    "Decision Tree": "#2ca02c",
    "Naive Bayes": "#9467bd",
    "KNN": "#ff7f0e",
    "Random Forest": "#8c564b",
    "Gradient Boosting": "#17becf",
    "AdaBoost": "#e377c2",
    "XGBoost": "#E53D00",
    "LightGBM": "#FFCF00",
    "CatBoost": "#8AFFC1"
}

results = []
numberofsplits = 5
skf = StratifiedKFold(n_splits=numberofsplits, shuffle=True, random_state=42)

for model_name, model in models.items():
    cv_accuracies = []
    cv_auc_scores = []
    cv_f1_scores = []
    cv_precisions = []
    cv_recalls = []

    for train_index, val_index in skf.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_index], X_train[val_index]
        y_train_cv, y_val_cv = y_train.to_numpy()[train_index], y_train.to_numpy()[val_index]

        model.fit(X_train_cv, y_train_cv)
        y_val_pred = model.predict(X_val_cv)
        y_val_pred_proba = model.predict_proba(X_val_cv)[:, 1] if hasattr(model, 'predict_proba') else None

        cv_accuracies.append(accuracy_score(y_val_cv, y_val_pred))
        cv_f1_scores.append(f1_score(y_val_cv, y_val_pred))
        cv_precisions.append(precision_score(y_val_cv, y_val_pred))
        cv_recalls.append(recall_score(y_val_cv, y_val_pred))

        if y_val_pred_proba is not None:
            cv_auc_scores.append(roc_auc_score(y_val_cv, y_val_pred_proba))

    results.append({
        "Model": model_name,
        "Accuracy": np.mean(cv_accuracies),
        "AUC": np.mean(cv_auc_scores) if cv_auc_scores else None,
        "F1 Score": np.mean(cv_f1_scores),
        "Precision": np.mean(cv_precisions),
        "Recall": np.mean(cv_recalls)
    })

results_df = pd.DataFrame(results)
metrics = ["Accuracy", "AUC", "F1 Score", "Precision", "Recall"]

# Plot metrics for each model on the same chart
for _, row in results_df.iterrows():
    metrics_values = [row["Accuracy"], row["AUC"], row["F1 Score"], row["Precision"], row["Recall"]]
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, metrics_values, color=model_colors[row['Model']])

    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha='center', fontsize=10)

    plt.title(f"Metrics for {row['Model']}", fontsize=14)
    plt.xlabel("Metrics", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.ylim(0.6, 0.85)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"wykresy/wykresyModele/Comparison_{row['Model']}_metrics.png", dpi=300)
    plt.show()

# Plot metrics comparison across models
for metric in metrics:
    if metric in results_df.columns and results_df[metric].notna().any():
        plt.figure(figsize=(12, 8))
        sorted_results = results_df.sort_values(by=metric, ascending=False)
        ax = sns.barplot(
            x=metric, y="Model",
            data=sorted_results,
            palette=[model_colors[model] for model in sorted_results["Model"]]
        )
        plt.title(f"Comparison of Models by {metric} ({numberofsplits} splits)", fontsize=16)
        plt.xlabel(metric, fontsize=14)
        plt.ylabel("Model", fontsize=14)
        plt.xlim(0.6, 0.85)

        # Add values on the bars
        for i, (value, model) in enumerate(zip(sorted_results[metric], sorted_results["Model"])):
            ax.text(value + 0.005, i, f"{value:.3f}", color='black', va='center', fontsize=12)

        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"wykresy/wykresyModele/Comparison_{metric}_all_models.png", dpi=300)
        plt.show()
