import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.utils import resample
from scipy.stats import mannwhitneyu
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from collections import defaultdict, Counter

X = pd.read_csv("Introduce path to csv file containing radiomic features")
y = np.array([0,1,0,0,1,1,0,1,0,1,1,1,1,1,0,1,1,1,0,1,1,0,1,0,0,0,1,1,1,1,1,0,0,1,1,0,0,0,1,1,1,0,1,0,1,1]) # Target labels

X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

def initial_feature_selection(X, y, var_thr=1e-4, spearman_thr=0.99, pval_thr=0.4):
    selector = VarianceThreshold(threshold=var_thr)
    X_var = pd.DataFrame(selector.fit_transform(X), columns=X.columns[selector.get_support()])
    corr_matrix = X_var.corr(method='spearman')
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(abs(upper_tri[col]) > spearman_thr)]
    X_uncorr = X_var.drop(columns=to_drop)
    selected_features = [f for f in X_uncorr.columns if mannwhitneyu(X_uncorr[f][y==0], X_uncorr[f][y==1])[1] < pval_thr]
    return X_uncorr[selected_features]

models = {
    "ElasticNet": LogisticRegression(penalty='elasticnet', solver='saga', max_iter=10000, class_weight='balanced', l1_ratio=0.1, C=0.1),
    "XGBoost": XGBClassifier( eval_metric='logloss'),
    "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
    "SVM_Linear": LinearSVC(class_weight='balanced', max_iter=10000, C=0.01)
}

N_BOOTSTRAPS = 10
TOP_N = 20
STABILITY_THRESHOLD = 4
feature_counts = {model: defaultdict(int) for model in models}

for i in range(N_BOOTSTRAPS):
    print(f"\n Bootstrapping iteration {i+1}/{N_BOOTSTRAPS}")
    X_res, y_res = resample(X_trainval, y_trainval, stratify=y_trainval, random_state=i)
    X_filtered = initial_feature_selection(X_res, y_res)
    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_filtered), columns=X_filtered.columns)

    for name, model in models.items():
        try:
            model.fit(X_scaled, y_res)
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_.flatten())
            elif hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            else:
                continue
            sorted_idx = np.argsort(importances)[::-1]
            top_feats = list(X_scaled.columns[sorted_idx][:TOP_N])
            for feat in top_feats:
                feature_counts[name][feat] += 1
        except Exception as e:
            print(f" Error with model {name}: {e}")

rows = []
for model_name, counts in feature_counts.items():
    for feature, freq in counts.items():
        rows.append({"Modelo": model_name, "Variable": feature, "Frecuencia": freq})

df_freq = pd.DataFrame(rows)
df_freq = df_freq.pivot_table(index="Variable", columns="Modelo", values="Frecuencia", fill_value=0).reset_index()
binary_frequency = df_freq.drop(columns="Variable") > 0  
num_models_by_variable = binary_frequency.sum(axis=1)
df_freq_filtered = df_freq[num_models_by_variable >= 2]

selected_by_2_models = df_freq_filtered.drop(columns="Variable") > 2
at_least_one_model_with_high_freq = selected_by_2_models.any(axis=1)
df_freq_filtered = df_freq_filtered[at_least_one_model_with_high_freq]

df_long = df_freq_filtered.melt(id_vars="Variable", var_name="Model", value_name="Frequency")

for name in models:
    print(f"\n Frecuency of appearance in top-{TOP_N} for model {name}:")
    most_common = Counter(feature_counts[name]).most_common(20)
    for feat, count in most_common:
        print(f"{feat}: {count}/{N_BOOTSTRAPS}")

print("\n Evaluation using stable features (≥4/10 appearances in top-N):")
for name, model in models.items():
    freq = Counter(feature_counts[name])
    stable_feats = [f for f, c in freq.items() if c >= STABILITY_THRESHOLD]
    if len(stable_feats) < 5:
        stable_feats = [f for f, _ in most_common[:5]]  
        
    if not stable_feats:
        print(f"\n {name} does not have sufficient stable features.")
        continue

    print(f"\n {name} - {len(stable_feats)} stable features")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    pipeline.fit(X_trainval[stable_feats], y_trainval)
    y_pred = pipeline.predict(X_test[stable_feats])
    y_proba = pipeline.predict_proba(X_test[stable_feats])[:, 1] if hasattr(pipeline.named_steps['model'], 'predict_proba') else pipeline.decision_function(X_test[stable_feats])

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap='Blues')
    plt.title(f"Matriz de Confusión - {name}")
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(12, 10))
sns.barplot(data=df_long, x="Variable", y="Frequency", hue="Model")
plt.axhline(5, color='red', linestyle='--', label='Stability threshold (5)')
plt.title("Frequency of appearance in bootstrapping by variable and model")
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# Common feature set
def evaluate_with_stable_common_features(feature_counts_by_model, 
                                         X_trainval, X_test, y_trainval, y_test, 
                                         models_dict, 
                                         min_freq=5, min_models=2, top_k_fallback=5):
    print(f"\n Searching for features selected ≥{min_freq} times in ≥{min_models} models...\n")

    feature_model_map = {}
    
    for model, feats in feature_counts_by_model.items():
        for feat, freq in feats.items():
            if freq >= min_freq:
                feature_model_map.setdefault(feat, set()).add(model)

    robust_features = [f for f, models in feature_model_map.items() if len(models) >= min_models]

    if not robust_features:
        print(" No features sufficiently stable were found. Using fallback.")
        # Fallback: use top-k more frequent in total
        global_counts = Counter()
        for model_feats in feature_counts_by_model.values():
            global_counts.update(model_feats)
        robust_features = [f for f, _ in global_counts.most_common(top_k_fallback)]

    print(f"✅ {len(robust_features)} features selected:")
    for f in robust_features:
        print(f"  {f}  (modelos: {', '.join(feature_model_map.get(f, []))})")

    print("\n Evaluation of models with common features:")

    for name, model in models_dict.items():
        print(f"\n🔧 Modelo: {name}")
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])

        pipeline.fit(X_trainval[robust_features], y_trainval)
        y_pred = pipeline.predict(X_test[robust_features])
        y_proba = pipeline.predict_proba(X_test[robust_features])[:, 1] if hasattr(pipeline.named_steps['model'], 'predict_proba') else pipeline.decision_function(X_test[robust_features])

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)

        print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")

        cm = confusion_matrix(y_test, y_pred)
        ConfusionMatrixDisplay(cm).plot(cmap='Blues')
        plt.title(f"Confusion Matrix - {name}")
        plt.tight_layout()
        plt.show()
    return robust_features

robust_features=evaluate_with_stable_common_features(
    feature_counts_by_model=feature_counts,
    X_trainval=X_trainval,
    X_test=X_test,
    y_trainval=y_trainval,
    y_test=y_test,
    models_dict=models, 
    min_freq=5,
    min_models=2
)

shap_means_subset = {}
# Visualización con features comunes seleccionados
for name, model in models.items():
    
    print(f"\n Visualization for model: {name}")
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    pipeline.fit(X_trainval[robust_features], y_trainval)

    # Predicciones y probabilidades
    y_pred = pipeline.predict(X_test[robust_features])
    if hasattr(pipeline.named_steps['model'], 'predict_proba'):
        y_proba = pipeline.predict_proba(X_test[robust_features])[:, 1]
    else:
        scores = pipeline.decision_function(X_test[robust_features])
        y_proba = expit(scores)

    # SHAP values
    print("🔍 Interpretability SHAP...")
    try:
        if name in ["XGBoost", "RandomForest"]:
            explainer = shap.Explainer(pipeline.named_steps['model'], X_trainval[robust_features])
        else:
            explainer = shap.Explainer(pipeline.named_steps['model'], X_trainval[robust_features], feature_names=robust_features)
        
        shap_values = explainer(X_test[robust_features])
        # Calcular media del valor absoluto SHAP para cada variable
        try:
            shap_vals_raw = shap_values.values
            if shap_vals_raw.ndim > 2:  # Clasificación binaria
                 shap_vals_raw = shap_vals_raw[..., 1]
            shap_mean_abs = np.abs(shap_vals_raw).mean(axis=0)
            shap_means_subset[name] = pd.Series(shap_mean_abs, index=robust_features)
        except Exception as e:
            print(f" Could not compute mean SHAP values for {name}: {e}")

        if hasattr(shap_values.values, 'ndim') and shap_values.values.ndim > 2:
            shap_values = shap.Explanation(values=shap_values.values[..., 1],
                                           base_values=shap_values.base_values[..., 1],
                                           data=shap_values.data,
                                           feature_names=shap_values.feature_names)

        shap.plots.beeswarm(shap_values, max_display=10, show=True)
        shap.plots.bar(shap_values, max_display=10, show=True)

    except Exception as e:
        print(f" Error in SHAP for {name}: {e}")

    # ROC Curve
    RocCurveDisplay.from_predictions(y_test, y_proba)
    plt.title(f"Curva ROC - {name}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Calibration curve
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
        print(f" Only one class present in test, calibration curve not applicable to {name}")

df_shap_subset = pd.DataFrame(shap_means_subset).fillna(0)
df_shap_subset = df_shap_subset.drop(columns=["XGBoost"], errors='ignore')

top_k = 10
top_vars_subset = df_shap_subset.mean(axis=1).nlargest(top_k).index
df_topk_subset = df_shap_subset.loc[top_vars_subset]

df_long_subset = df_topk_subset.reset_index().melt(id_vars='index', var_name='Model', value_name='SHAP_mean')
df_long_subset = df_long_subset.rename(columns={"index": "Variable"})

plt.figure(figsize=(12, 6))
sns.barplot(data=df_long_subset, x="Variable", y="SHAP_mean", hue="Model")
plt.xticks(rotation=45, ha='right')
plt.ylabel("Mean SHAP value (|impact|)")
plt.title("Mean SHAP importance by variable and model (Top features)")
plt.legend(title="Model")
plt.tight_layout()
plt.show()
