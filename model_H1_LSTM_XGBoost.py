# =============================================================================
#  MODEL H1  : LSTM  →  XGBoost  (Hybrid Deep-Learning + Gradient Boosting)
#  Strategy : DL first  →  extract features  →  XGBoost classifier
#  Dataset  : UNSW-NB15
#  Author   : Master SIDI – FST Errachidia  |  Resp. F. AMOUNAS
#  Env      : VS Code / Google Colab (Python 3.9+)
# =============================================================================

# ─────────────────────────────────────────────────────────────────────────────
#  HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
LEARNING_RATE     = 0.001
EPOCHS            = 30
BATCH_SIZE        = 256
TEST_SIZE         = 0.20
RANDOM_STATE      = 42
K_BEST_FEATURES   = 25
LSTM_UNITS        = 128        # slightly bigger for richer features
FEATURE_VECTOR_DIM= 32         # larger feature vector → richer for XGB
DROPOUT_RATE      = 0.3
PATIENCE          = 5
SMOTE_SAMPLING    = "auto"

# XGBoost hyperparameters
XGB_N_ESTIMATORS  = 300
XGB_MAX_DEPTH     = 6
XGB_LEARNING_RATE = 0.05       # XGBoost's own lr (different from DL)
XGB_SUBSAMPLE     = 0.8
XGB_COLSAMPLE     = 0.8
XGB_USE_GPU       = False      # set True if CUDA available

# ─────────────────────────────────────────────────────────────────────────────
#  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, warnings, time
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing     import LabelEncoder, StandardScaler
from sklearn.impute             import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics           import (classification_report,
                                        confusion_matrix,
                                        precision_recall_curve,
                                        average_precision_score,
                                        accuracy_score,
                                        precision_score, recall_score,
                                        f1_score)
from imblearn.over_sampling    import SMOTE
import joblib

import tensorflow as tf
from tensorflow.keras.models   import Model
from tensorflow.keras.layers   import (Input, LSTM, Dense,
                                        Dropout, BatchNormalization,
                                        Bidirectional)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau

from xgboost import XGBClassifier

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV  = "UNSW_NB15_testing-set.csv"
IMG_DIR   = "images_H1"
MODEL_DIR = "models_H1"
os.makedirs(IMG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  MODEL H1 : LSTM + XGBoost  –  UNSW-NB15")
print("=" * 65)

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
print(f"Train : {df_train.shape}  |  Test : {df_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
#  2. PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(df_tr, df_te):
    drop_cols = ["id"] if "id" in df_tr.columns else []
    df_tr = df_tr.drop(columns=drop_cols, errors="ignore")
    df_te = df_te.drop(columns=drop_cols, errors="ignore")

    le = LabelEncoder()
    for col in df_tr.select_dtypes(include="object").columns:
        if col == "attack_cat":
            continue
        df_tr[col] = le.fit_transform(df_tr[col].astype(str))
        df_te[col] = le.transform(df_te[col].astype(str).map(
            lambda x: x if x in le.classes_ else le.classes_[0]))

    y_tr   = df_tr["label"].values
    y_te   = df_te["label"].values
    cat_tr = df_tr["attack_cat"].values if "attack_cat" in df_tr.columns else None

    feat_cols = [c for c in df_tr.columns if c not in ("label", "attack_cat")]
    X_tr = df_tr[feat_cols].values.astype(np.float32)
    X_te = df_te[feat_cols].values.astype(np.float32)

    imp  = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    sel  = SelectKBest(f_classif, k=min(K_BEST_FEATURES, X_tr.shape[1]))
    X_tr = sel.fit_transform(X_tr, y_tr)
    X_te = sel.transform(X_te)
    selected = [feat_cols[i] for i in sel.get_support(indices=True)]
    print(f"[FeatureSelect] {len(selected)} features: {selected}")

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    return X_tr, X_te, y_tr, y_te, cat_tr, selected


X_train, X_test, y_train, y_test, cat_train, feat_names = \
    preprocess(df_train.copy(), df_test.copy())

# ─────────────────────────────────────────────────────────────────────────────
#  3. PLOTS – Class & Attack distribution
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
u, c = np.unique(y_train, return_counts=True)
plt.bar([str(x) for x in u], c, color=["steelblue","tomato"])
plt.title("Class Distribution (Train) – H1")
plt.xlabel("Label (0=Normal, 1=Attack)"); plt.ylabel("Count")
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/01_class_distribution.png", dpi=150); plt.close()

if cat_train is not None:
    plt.figure(figsize=(10, 5))
    pd.Series(cat_train).value_counts().plot(kind="bar", color="steelblue", edgecolor="k")
    plt.title("Attack Types – H1"); plt.xticks(rotation=45, ha="right")
    plt.tight_layout(); plt.savefig(f"{IMG_DIR}/02_attack_types.png", dpi=150); plt.close()

print("[Plot] Distributions saved.")

# ─────────────────────────────────────────────────────────────────────────────
#  4. SMOTE
# ─────────────────────────────────────────────────────────────────────────────
print("[SMOTE] Balancing …")
sm = SMOTE(sampling_strategy=SMOTE_SAMPLING, random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"[SMOTE] {X_train_sm.shape} | {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ─────────────────────────────────────────────────────────────────────────────
#  5. RESHAPE
# ─────────────────────────────────────────────────────────────────────────────
X_tr3 = X_train_sm.reshape(-1, 1, X_train_sm.shape[1])
X_te3 = X_test.reshape(-1, 1, X_test.shape[1])

# ─────────────────────────────────────────────────────────────────────────────
#  6. LSTM FEATURE EXTRACTOR  (stacked, with return_sequences → attention-like)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LSTM] Building model …")
n_feat = X_tr3.shape[2]
inp    = Input(shape=(1, n_feat))
x      = LSTM(LSTM_UNITS, return_sequences=True)(inp)
x      = Dropout(DROPOUT_RATE)(x)
x      = LSTM(LSTM_UNITS // 2, return_sequences=False)(x)
x      = BatchNormalization()(x)
x      = Dropout(DROPOUT_RATE)(x)
feat_o = Dense(FEATURE_VECTOR_DIM, activation="relu", name="feature_vector")(x)
clf_o  = Dense(1, activation="sigmoid")(feat_o)

lstm_model = Model(inp, clf_o, name="LSTM_H1")
lstm_model.compile(Adam(LEARNING_RATE), loss="binary_crossentropy",
                   metrics=["accuracy"])
lstm_model.summary()

cbs = [
    EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

print("\n[LSTM] Training …")
t0 = time.time()
hist = lstm_model.fit(
    X_tr3, y_train_sm,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=TEST_SIZE,
    callbacks=cbs, verbose=1
)
lstm_time = time.time() - t0

# ─────────────────────────────────────────────────────────────────────────────
#  7. LOSS / ACCURACY CURVES
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(hist.history["loss"],     label="Train")
ax[0].plot(hist.history["val_loss"], label="Val")
ax[0].set_title("LSTM Loss – H1"); ax[0].legend()
ax[1].plot(hist.history["accuracy"],     label="Train")
ax[1].plot(hist.history["val_accuracy"], label="Val")
ax[1].set_title("LSTM Accuracy – H1"); ax[1].legend()
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/03_loss_accuracy.png", dpi=150); plt.close()
print("[Plot] Loss/Accuracy saved.")

# ─────────────────────────────────────────────────────────────────────────────
#  8. EXTRACT FEATURES
# ─────────────────────────────────────────────────────────────────────────────
fe = Model(inp, lstm_model.get_layer("feature_vector").output, name="FE_H1")
X_tr_feat = fe.predict(X_tr3, batch_size=BATCH_SIZE, verbose=0)
X_te_feat = fe.predict(X_te3, batch_size=BATCH_SIZE, verbose=0)
print(f"[Features] Train={X_tr_feat.shape}  Test={X_te_feat.shape}")

# ─────────────────────────────────────────────────────────────────────────────
#  9. XGBoost CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────
tree_method = "gpu_hist" if XGB_USE_GPU else "hist"

xgb = XGBClassifier(
    n_estimators    = XGB_N_ESTIMATORS,
    max_depth       = XGB_MAX_DEPTH,
    learning_rate   = XGB_LEARNING_RATE,
    subsample       = XGB_SUBSAMPLE,
    colsample_bytree= XGB_COLSAMPLE,
    tree_method     = tree_method,
    use_label_encoder=False,
    eval_metric     = "logloss",
    random_state    = RANDOM_STATE,
    n_jobs          = -1,
    scale_pos_weight= (y_train_sm == 0).sum() / (y_train_sm == 1).sum()
)

print("\n[XGBoost] Training on LSTM feature vectors …")
t0 = time.time()
xgb.fit(
    X_tr_feat, y_train_sm,
    eval_set=[(X_te_feat, y_test)],
    verbose=50
)
xgb_time   = time.time() - t0
total_time  = lstm_time + xgb_time

y_pred  = xgb.predict(X_te_feat)
y_proba = xgb.predict_proba(X_te_feat)[:, 1]
acc     = accuracy_score(y_test, y_pred)
report  = classification_report(y_test, y_pred,
                                 target_names=["Normal","Attack"], digits=4)

print("\n" + "=" * 65)
print("  CLASSIFICATION REPORT – Model H1 (LSTM + XGBoost)")
print("=" * 65)
print(report)
print(f"  Accuracy : {acc:.4f}  |  Time : {total_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
#  10. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges",
            xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
plt.title("Confusion Matrix – H1 (LSTM + XGBoost)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/04_confusion_matrix.png", dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  11. FEATURE IMPORTANCE (XGBoost on LSTM features)
# ─────────────────────────────────────────────────────────────────────────────
imp        = xgb.feature_importances_
labels     = [f"LSTM_feat_{i}" for i in range(FEATURE_VECTOR_DIM)]
sorted_idx = np.argsort(imp)[::-1]
plt.figure(figsize=(8, 5))
plt.bar(range(FEATURE_VECTOR_DIM), imp[sorted_idx], color="darkorange", edgecolor="k")
plt.xticks(range(FEATURE_VECTOR_DIM), [labels[i] for i in sorted_idx], rotation=90)
plt.title("Feature Importance – H1 (XGBoost on LSTM features)")
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/05_feature_importance.png", dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  12. PRECISION-RECALL CURVE
# ─────────────────────────────────────────────────────────────────────────────
prec, rec, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(rec, prec, color="darkorange", lw=2, label=f"AP={ap:.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve – H1 (LSTM + XGBoost)")
plt.legend(); plt.tight_layout()
plt.savefig(f"{IMG_DIR}/06_precision_recall_curve.png", dpi=150); plt.close()
print("[Plot] All plots saved.")

# ─────────────────────────────────────────────────────────────────────────────
#  13. RESULTS FILE
# ─────────────────────────────────────────────────────────────────────────────
prec_w = precision_score(y_test, y_pred, average="weighted")
rec_w  = recall_score   (y_test, y_pred, average="weighted")
f1_w   = f1_score       (y_test, y_pred, average="weighted")
prec_a = precision_score(y_test, y_pred, pos_label=1)
rec_a  = recall_score   (y_test, y_pred, pos_label=1)
f1_a   = f1_score       (y_test, y_pred, pos_label=1)
prec_n = precision_score(y_test, y_pred, pos_label=0)
rec_n  = recall_score   (y_test, y_pred, pos_label=0)
f1_n   = f1_score       (y_test, y_pred, pos_label=0)

txt = f"""
============================================================
  RESULTS – Model H1 : LSTM + XGBoost (UNSW-NB15)
============================================================
Accuracy       : {acc:.4f}
Precision (W)  : {prec_w:.4f}
Recall    (W)  : {rec_w:.4f}
F1-Score  (W)  : {f1_w:.4f}
Training time  : {total_time:.1f} s
Average Prec.  : {ap:.4f}

-- With Anomaly (Attack=1) --
Precision : {prec_a:.4f}  |  Recall : {rec_a:.4f}  |  F1 : {f1_a:.4f}

-- Without Anomaly (Normal=0) --
Precision : {prec_n:.4f}  |  Recall : {rec_n:.4f}  |  F1 : {f1_n:.4f}

------------------------------------------------------------
Selected Features (top {K_BEST_FEATURES}):
{feat_names}
------------------------------------------------------------
LSTM feature vector dim : {FEATURE_VECTOR_DIM}
SMOTE applied           : Yes
"""
with open(f"{IMG_DIR}/results_H1.txt", "w") as f:
    f.write(txt)
print("[Results] Saved."); print(txt)

# ─────────────────────────────────────────────────────────────────────────────
#  14. SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────
lstm_model.save(f"{MODEL_DIR}/lstm_model_H1.h5")
fe.save(f"{MODEL_DIR}/feature_extractor_H1.h5")
joblib.dump(xgb, f"{MODEL_DIR}/xgboost_H1.joblib")
print(f"[Save] Models saved in '{MODEL_DIR}/'\nDone – Model H1.")
