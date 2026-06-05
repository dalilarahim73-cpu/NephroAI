# ============================================================
#  train.py  —  ESRD Prediction  —  XGBoost Pipeline
#  Compatible Render.com (Free Tier)
#  - Cherche le CSV dans data/ ou à la racine
#  - Si introuvable : génère un dataset synthétique minimal
#    pour créer un pipeline.pkl fonctionnel (fallback)
# ============================================================
import os
import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

print("=" * 60)
print("  ESRD PREDICTION — TRAINING PIPELINE (XGBoost)")
print("=" * 60)

# ── 1. Chargement du dataset ──────────────────────────────────
# Cherche dans plusieurs chemins possibles
CSV_PATHS = [
    "data/esrd_prediction_dataset.csv",
    "esrd_prediction_dataset.csv",
    os.path.join(os.path.dirname(__file__), "data", "esrd_prediction_dataset.csv"),
    os.path.join(os.path.dirname(__file__), "esrd_prediction_dataset.csv"),
]

data = None
for path in CSV_PATHS:
    if os.path.exists(path):
        data = pd.read_csv(path)
        print(f"[OK] Dataset chargé depuis : {path}")
        print(f"     Shape : {data.shape}")
        break

if data is None:
    print("[WARN] Dataset CSV introuvable — génération d'un dataset synthétique minimal")
    print("       Le modèle utilisera le fallback heuristique dans App.py")
    # Crée un pipeline minimal fonctionnel pour que l'import joblib réussisse
    np.random.seed(42)
    n = 500
    feature_cols_synth = [
        'Age', 'Baseline Serum Creatinine (mg/dL)', 'Mean Serum Creatinine (mg/dL)',
        'Cholesterol (mg/dL)', 'Triglyceride (mg/dL)', 'LDL-C (mg/dL)', 'HDL-C (mg/dL)',
        'Uric Acid (mg/dL)', 'Calcium (mg/dL)', 'Phosphate (mg/dL)', 'Hemoglobin (g/dL)',
        'Albumin (g/dL)', 'HS-CRP (mg/dL)', 'HbA1c (%)', 'Glucose (mg/dL)',
        'Gender', 'Smoking', 'Alcohol', 'Hypertension', 'Coronary Artery Disease',
        'Cancer', 'Chronic Liver Disease', 'Diabetic Retinopathy',
        'NSAID', 'Statin', 'Metformin', 'Insulin', 'Dipeptidyl Peptidase-4 Inhibitor'
    ]
    synth = pd.DataFrame(np.random.randn(n, len(feature_cols_synth)), columns=feature_cols_synth)
    # Variable cible basée sur créatinine et âge (règle simple)
    y_synth = ((synth['Age'] > 0.5) & (synth['Baseline Serum Creatinine (mg/dL)'] > 0.5)).astype(int)

    imputer_s = SimpleImputer(strategy='median')
    scaler_s  = StandardScaler()
    X_s = imputer_s.fit_transform(synth)
    X_s = scaler_s.fit_transform(X_s)

    model_s = XGBClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        eval_metric='auc', n_jobs=-1, random_state=42, tree_method='hist'
    )
    model_s.fit(X_s, y_synth)

    pipeline = {
        'model':       model_s,
        'imputer':     imputer_s,
        'scaler':      scaler_s,
        'features':    feature_cols_synth,
        'cat_cols':    [
            "Gender", "Smoking", "Alcohol", "Hypertension",
            "Coronary Artery Disease", "Cancer", "Chronic Liver Disease",
            "Diabetic Retinopathy", "NSAID", "Statin", "Metformin",
            "Insulin", "Dipeptidyl Peptidase-4 Inhibitor"
        ],
        'threshold':   0.5,
        'label_names': ['No ESRD Risk', 'ESRD Risk'],
        'model_name':  'XGBoost (synthetic fallback)',
        'n_features':  len(feature_cols_synth),
    }
    joblib.dump(pipeline, 'esrd_pipeline.pkl')
    print("[OK] Pipeline synthétique sauvegardé : esrd_pipeline.pkl")
    print("=" * 60)
    exit(0)

# ── 2. Encodage catégoriel ────────────────────────────────────
CAT_COLS = [
    "Gender", "Smoking", "Alcohol", "Hypertension",
    "Coronary Artery Disease", "Cancer", "Chronic Liver Disease",
    "Diabetic Retinopathy", "NSAID", "Statin", "Metformin",
    "Insulin", "Dipeptidyl Peptidase-4 Inhibitor"
]
le = LabelEncoder()
for col in CAT_COLS:
    if col in data.columns:
        data[col] = le.fit_transform(data[col].astype(str))

data['class'] = (data['ESRD Risk'] == 'Yes').astype(int)

# ── 3. Split train / test ─────────────────────────────────────
data = data.reset_index(drop=True)
META_COLS    = ['Patient ID', 'Dataset Split', 'ESRD Risk', 'class']
feature_cols = [c for c in data.columns
                if c not in META_COLS
                and pd.api.types.is_numeric_dtype(data[c])]

if 'Dataset Split' in data.columns:
    train_idx = data.index[data['Dataset Split'] == 'Training'].tolist()
    test_idx  = data.index[data['Dataset Split'] == 'Testing'].tolist()
else:
    from sklearn.model_selection import train_test_split
    idx = list(range(len(data)))
    train_idx, test_idx = train_test_split(idx, test_size=0.3, random_state=42)

print(f"Train: {len(train_idx)} rows  |  Test: {len(test_idx)} rows")

X_raw = data[feature_cols]
y     = data['class'].values

# ── 4. Imputation + Scaling ───────────────────────────────────
imputer = SimpleImputer(strategy='median')
X_tr    = imputer.fit_transform(X_raw.iloc[train_idx])
X_te    = imputer.transform(X_raw.iloc[test_idx])

scaler  = StandardScaler()
X_tr    = scaler.fit_transform(X_tr)
X_te    = scaler.transform(X_te)

y_train = y[train_idx]
y_test  = y[test_idx]

n_neg = int(np.sum(y_train == 0))
n_pos = int(np.sum(y_train == 1))
spw   = round(n_neg / max(n_pos, 1))
print(f"Class balance — No: {n_neg}  Yes: {n_pos}  → scale_pos_weight={spw}\n")

# ── 5. Entraînement XGBoost ───────────────────────────────────
print("Training XGBoost...")
model = XGBClassifier(
    n_estimators     = 200,
    max_depth        = 5,
    learning_rate    = 0.1,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    scale_pos_weight = spw,
    eval_metric      = 'auc',
    n_jobs           = -1,
    random_state     = 42,
    tree_method      = 'hist'
)
model.fit(X_tr, y_train, eval_set=[(X_te, y_test)], verbose=False)

# ── 6. Évaluation ─────────────────────────────────────────────
probs  = model.predict_proba(X_te)[:, 1]
y_pred = model.predict(X_te)

print("\n" + "=" * 60)
print("  RÉSULTATS SUR LE TEST SET")
print("=" * 60)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-Score  : {f1_score(y_test, y_pred, average='weighted'):.4f}")
try:
    print(f"AUC-ROC   : {roc_auc_score(y_test, probs):.4f}")
except Exception:
    pass

# ── 7. Sauvegarde du pipeline ─────────────────────────────────
pipeline = {
    'model':       model,
    'imputer':     imputer,
    'scaler':      scaler,
    'features':    feature_cols,
    'cat_cols':    CAT_COLS,
    'threshold':   0.5,
    'label_names': ['No ESRD Risk', 'ESRD Risk'],
    'model_name':  'XGBoost',
    'n_features':  len(feature_cols),
}
joblib.dump(pipeline, 'esrd_pipeline.pkl')
print("=" * 60)
print("  ✅ Pipeline sauvegardé : esrd_pipeline.pkl")
print("=" * 60)
