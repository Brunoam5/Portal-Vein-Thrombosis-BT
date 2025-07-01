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

# Cargar solo las variables clínicas
X_clinical = pd.read_csv("/Users/bruno/Desktop/TFG/03_Radiomics/ThrombusMasks v3/info_clinica_y_manual.csv")
y = np.array([0,1,0,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,1])

# Separar en entrenamiento y test
X = X_clinical.copy()
clinical_cols = X.columns.tolist()
os.makedirs("resultados_clinicos", exist_ok=True)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(X_trainval.columns)

# Test de Mann–Whitney U por variable clínica
print("\n📊 Análisis de significancia estadística (Mann–Whitney U):")
significativos = []
pvals = []

for col in clinical_cols:
    grupo_0 = X_trainval[y_trainval == 0][col]
    grupo_1 = X_trainval[y_trainval == 1][col]

    try:
        stat, p = mannwhitneyu(grupo_0, grupo_1, alternative='two-sided')
        pvals.append((col, p))
        if p < 0.05:
            print(f"  ✅ {col}: p = {p:.4f} (significativo)")
            significativos.append(col)
        else:
            print(f"  {col}: p = {p:.4f}")
    except ValueError:
        print(f"⚠️ {col}: no se pudo calcular (posible variable binaria constante o nula)")
        
# Filtrado inicial
def initial_feature_selection(X, y, pval_thr=0.4):
    selected_features = [f for f in X.columns if mannwhitneyu(X[f][y==0], X[f][y==1])[1] < pval_thr]
    
    return X[selected_features]

#X_trainval=initial_feature_selection(X_trainval, y_trainval)

#print(X_trainval)


# Definir modelos
models = {
    "ElasticNet": LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, class_weight='balanced', l1_ratio=0.1, C=0.1),
    #"Lasso": LogisticRegression(penalty='l1', solver='saga', max_iter=10000, class_weight='balanced', C=1),
    "XGBoost": XGBClassifier(eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM_Linear": LinearSVC(class_weight='balanced', max_iter=10000, C=0.01)
}


shap_means = {}
perm_means = {}

# Evaluación por modelo
for name, model in models.items():
    print(f"\n Modelo clínico: {name}")
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
           y_proba = expit(scores)  # convierte scores a "probabilidades" entre 0 y 1

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        plt.title(f"Matriz de Confusión - {name}")
        plt.tight_layout()
        plt.show()

        # SHAP
        print("🔍 Interpretabilidad SHAP...")
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
        
        # Guardar SHAP mean values por modelo
        try:
            shap_vals_raw = shap_values.values
            if shap_vals_raw.ndim > 2:  # Clasificación binaria con output 2D
               shap_vals_raw = shap_vals_raw[..., 1]
            shap_mean_abs = np.abs(shap_vals_raw).mean(axis=0)
            shap_means[name] = pd.Series(shap_mean_abs, index=clinical_cols)

        except Exception as e:
            print(f"⚠️ No se pudo calcular SHAP mean values para {name}: {e}")
        
        # Permutation importance
        print("🔁 Calculando Permutation Importance...")
        try:
            result = permutation_importance(
               pipeline, X_test[clinical_cols], y_test,
               n_repeats=10, random_state=42, scoring='roc_auc'
            )
            importances = result.importances_mean
            perm_means[name] = pd.Series(importances, index=clinical_cols)
        except Exception as e_perm:
            print(f"⚠️ Error en permutation importance para {name}: {e_perm}")


        # Histograma con seaborn (variable ejemplo: Oclusion_RightPV)
        if "Oclusion_RightPV" in clinical_cols:
            X_test_copy = X_test.copy()
            X_test_copy["Respuesta"] = y_test
            plt.figure()
            sns.histplot(data=X_test_copy, x="Oclusion_RightPV", hue="Respuesta", multiple="stack", palette="coolwarm")
            plt.title("Distribución de Oclusión Right PV según respuesta")
            plt.tight_layout()
            plt.show()

        # Curva ROC
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"Curva ROC - {name}")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Curva de calibración
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
            print(f"⚠️ Solo una clase en test, curva de calibración no aplicable para {name}")

    except Exception as e:
        print(f"⚠️ Error en modelo {name}: {e}")

# Combinar los shap_means en un DataFrame
df_shap = pd.DataFrame(shap_means).fillna(0)

# Opcional: seleccionar top-k variables más importantes en media
top_k = 10
top_vars = df_shap.mean(axis=1).nlargest(top_k).index
df_topk = df_shap.loc[top_vars]

# Convertir a formato largo para seaborn
df_long = df_topk.reset_index().melt(id_vars='index', var_name='Model', value_name='SHAP_mean')
df_long = df_long.rename(columns={"index": "Variable"})

# Gráfico de barras agrupadas por variable
plt.figure(figsize=(12, 6))
sns.barplot(data=df_long, x="Variable", y="SHAP_mean", hue="Model")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean SHAP value (|impact|)")
plt.title("Mean SHAP importance by variable and model (Demographic/manual subset)")
plt.legend(title="Model")
plt.tight_layout()
plt.show()

# Combinar resultados de permutation
df_perm = pd.DataFrame(perm_means).fillna(0)

# Top variables por media global
top_k_perm = 10
top_vars_perm = df_perm.mean(axis=1).nlargest(top_k_perm).index
df_topk_perm = df_perm.loc[top_vars_perm]

# Formato largo para seaborn
df_long_perm = df_topk_perm.reset_index().melt(id_vars='index', var_name='Modelo', value_name='Permutation_mean')
df_long_perm = df_long_perm.rename(columns={"index": "Variable"})

# Gráfico de barras agrupadas
plt.figure(figsize=(12, 6))
sns.barplot(data=df_long_perm, x="Variable", y="Permutation_mean", hue="Modelo")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Importancia media por Permutación (AUC)")
plt.title("Permutation Importance por variable y modelo (Top variables)")
plt.legend(title="Modelo")
plt.tight_layout()
plt.show()
