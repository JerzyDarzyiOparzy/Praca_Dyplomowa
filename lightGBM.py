import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix, roc_curve
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from fpdf import FPDF
import optuna
import joblib

data_path = "diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
df = pd.read_csv(data_path)

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"]

numerical_features = ["BMI", "Age", "PhysHlth", "MentHlth", "GenHlth", "Education", "Income"]
categorical_features = [col for col in X.columns if col not in numerical_features]

numerical_features2 = ["Age", "GenHlth", "Education", "Income"]
dni = ["PhysHlth", "MentHlth"]
bmi = ["BMI"]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("bmi", StandardScaler(), bmi),
        ("num", MinMaxScaler(), numerical_features2),
        ("rob", RobustScaler(), dni),
        ("cat", OneHotEncoder(drop='first'), categorical_features)
    ]
)
X_preprocessed = preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_preprocessed, y, test_size=0.2, random_state=42, stratify=y)


# Funkcja optymalizacyjna Optuna
def objective(trial):
    param = {
        'num_leaves': trial.suggest_int('num_leaves', 20, 50),
        'max_depth': trial.suggest_int('max_depth', -1, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.8, 1.0),
        'random_state': 42
    }
    model = LGBMClassifier(**param)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []

    for train_idx, val_idx in skf.split(X_train, y_train):
        X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
        y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
        model.fit(X_train_cv, y_train_cv)
        y_val_pred_proba = model.predict_proba(X_val_cv)[:, 1]
        aucs.append(roc_auc_score(y_val_cv, y_val_pred_proba))

    return np.mean(aucs)


# Optymalizacja parametrów
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50, n_jobs=-1)

best_params = study.best_params

numberofsplits = 5

best_lgbm = LGBMClassifier(**best_params)
skf = StratifiedKFold(n_splits=numberofsplits, shuffle=True, random_state=42)

final_accuracies = []
final_aucs = []
final_f1_scores = []
iteration_indices = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
    X_train_cv, X_val_cv = X_train[train_idx], X_train[val_idx]
    y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]

    best_lgbm.fit(X_train_cv, y_train_cv)
    y_val_pred = best_lgbm.predict(X_val_cv)
    y_val_pred_proba = best_lgbm.predict_proba(X_val_cv)[:, 1]

    final_accuracies.append(accuracy_score(y_val_cv, y_val_pred))
    final_aucs.append(roc_auc_score(y_val_cv, y_val_pred_proba))
    final_f1_scores.append(f1_score(y_val_cv, y_val_pred))
    iteration_indices.append(fold)

plt.figure(figsize=(12, 6), dpi=300)
plt.plot(iteration_indices, final_accuracies, marker='o', label='Accuracy')
plt.plot(iteration_indices, final_aucs, marker='o', label='AUC')
plt.plot(iteration_indices, final_f1_scores, marker='o', label='F1 Score')
plt.xticks(ticks=iteration_indices)
plt.title(f'Monitoring of Cross-Validation Metrics ({numberofsplits} splits)')
plt.xlabel('Cross-Validation Iteration')
plt.ylabel('Metric Value')
plt.legend()
plt.grid(alpha=0.7, linestyle='--')
plt.tight_layout()
plt.savefig("cross_validation_monitoring.png")
plt.close()

y_pred = best_lgbm.predict(X_test)
y_pred_proba = best_lgbm.predict_proba(X_test)[:, 1]

metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "AUC": roc_auc_score(y_test, y_pred_proba),
    "F1 Score": f1_score(y_test, y_pred)
}

# Wizualizacje
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {metrics["AUC"]:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.title('ROC Curve')
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

feature_importances = best_lgbm.feature_importances_
feature_names = preprocessor.get_feature_names_out()
sns.barplot(x=feature_importances, y=feature_names)
plt.title('Feature Importances')
plt.savefig("feature_importances.png")
plt.close()

# Zapis wytrenowanego modelu
joblib.dump(best_lgbm, "lightgbm_model.pkl")
joblib.dump(preprocessor, "preprocessor.pkl")


class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'LightGBM Model Report', align='C', ln=1)
        self.ln(10)

    def add_section(self, title, content):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, title, ln=1)
        self.ln(5)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 10, content)
        self.ln()

    def add_table(self, data):
        self.set_font('Arial', '', 10)
        col_width = self.w / 4
        row_height = self.font_size * 1.5
        for row in data:
            for item in row:
                self.cell(col_width, row_height, txt=str(item), border=1, ln=0, align='C')
            self.ln(row_height)

    def add_image(self, image_path):
        self.image(image_path, x=10, w=190)


pdf = PDF()
pdf.add_page()
pdf.add_section("Best Hyperparameters", "\n".join([f"{key}: {value}" for key, value in best_params.items()]))
pdf.add_section("Test Set Results", "\n".join([f"{key}: {value:.4f}" for key, value in metrics.items()]))
pdf.add_section("Classification Report", classification_report(y_test, y_pred))
pdf.add_page()
pdf.add_image("cross_validation_monitoring.png")
pdf.add_page()
pdf.add_image("confusion_matrix.png")
pdf.add_page()
pdf.add_image("roc_curve.png")
pdf.add_page()
pdf.add_image("feature_importances.png")

pdf.output("LightGBM_Model_Report.pdf")
