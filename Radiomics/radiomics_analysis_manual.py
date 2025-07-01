import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay)
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from scipy.special import expit 
from sklearn.inspection import permutation_importance
from scipy.stats import mannwhitneyu

X_manual = pd.read_csv("Introduce path to csv with manual data")
y = np.array([0,1,0,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,1])

# Separar en entrenamiento y test
X = X_manual.copy()
manual_cols = X.columns.tolist()

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print(X_trainval.columns)

# Test of Mann–Whitney U for each manual variable
print("\n Statistical significance analysis (Mann–Whitney U):")
significatives = []
pvals = []

for col in clinical_cols:
    group_0 = X_trainval[y_trainval == 0][col]
    group_1 = X_trainval[y_trainval == 1][col]

    try:
        stat, p = mannwhitneyu(group_0, group_1, alternative='two-sided')
        pvals.append((col, p))
        if p < 0.05:
            print(f"  ✅ {col}: p = {p:.4f} (significative)")
            significatives.append(col)
        else:
            print(f"  {col}: p = {p:.4f}")
    except ValueError:
        print(f"⚠️ {col}: could not compute (possible constant binary variable)")
        

models = {
    "ElasticNet": LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, class_weight='balanced', l1_ratio=0.1, C=0.1),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM_Linear": LinearSVC(class_weight='balanced', max_iter=10000, C=0.01)
}

shap_means = {}

for name, model in models.items():
    print(f"\n Manual data model: {name}")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    try:
        pipeline.fit(X_trainval[clinical_cols], y_trainval)
        y_pred = pipeline.predict(X_test[clinical_cols])
        if hasattr(model, 'predict_proba'):
           y_proba = pipeline.predict_proba(X_test[clinical_cols])[:, 1]
        else:
           scores = pipeline.decision_function(X_test[clinical_cols])
           y_proba = expit(scores)  

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        plt.title(f"Matriz de Confusión - {name}")
        plt.tight_layout()
        plt.show()

        print("🔍 Interpretability SHAP...")
        if name in ["XGBoost", "RandomForest"]:
            explainer = shap.Explainer(pipeline.named_steps['model'], X_trainval[clinical_cols])
        else:
            explainer = shap.Explainer(pipeline.named_steps['model'], X_trainval[clinical_cols], feature_names=clinical_cols)

        shap_values = explainer(X_test[clinical_cols])
        if hasattr(shap_values.values, 'ndim') and shap_values.values.ndim > 2:
                shap_values = shap.Explanation(values=shap_values.values[..., 1],
                                               base_values=shap_values.base_values[..., 1],
                                               data=shap_values.data,
                                               feature_names=shap_values.feature_names)

        shap.plots.beeswarm(shap_values, max_display=10, show=True)
        shap.plots.bar(shap_values, max_display=10, show=True)
        
        try:
            shap_vals_raw = shap_values.values
            if shap_vals_raw.ndim > 2:  
               shap_vals_raw = shap_vals_raw[..., 1]
            shap_mean_abs = np.abs(shap_vals_raw).mean(axis=0)
            shap_means[name] = pd.Series(shap_mean_abs, index=clinical_cols)

        except Exception as e:
            print(f"⚠️ No se pudo calcular SHAP mean values para {name}: {e}")

        # ROC Curve
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"Curva ROC - {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Calibration Curve
        if len(np.unique(y_test)) > 1:
            prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=5, strategy='uniform')
            plt.plot(prob_pred, prob_true, marker='o', label=name)
            plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
            plt.title("Curva de calibración")
            plt.xlabel("Probabilidad predicha")
            plt.ylabel("Probabilidad real")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print(f"⚠️ Only one class in test, calibration curve not applicable for {name}")

    except Exception as e:
        print(f"⚠️ Error in model {name}: {e}")


df_shap = pd.DataFrame(shap_means).fillna(0)

top_k = 10
top_vars = df_shap.mean(axis=1).nlargest(top_k).index
df_topk = df_shap.loc[top_vars]

df_long = df_topk.reset_index().melt(id_vars='index', var_name='Model', value_name='SHAP_mean')
df_long = df_long.rename(columns={"index": "Variable"})

plt.figure(figsize=(12, 6))
sns.barplot(data=df_long, x="Variable", y="SHAP_mean", hue="Model")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean SHAP value (|impact|)")
plt.title("Mean SHAP importance by variable and model (Demographic/manual subset)")
plt.legend(title="Model")
plt.tight_layout()
plt.show()
