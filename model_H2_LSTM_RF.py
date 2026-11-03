# =============================================================================
#  MODEL H2 : LSTM  →  Random Forest  (Hybrid Deep-Learning + ML)
#  Dataset  : UNSW-NB15
#  Author   : Master SIDI – FST Errachidia  |  Resp. F. AMOUNAS
#  Note     : Same architecture as A1 but uses UNIdirectional LSTM
#              → compare results with A1 (Bi-LSTM) in your report
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
LSTM_UNITS        = 64
FEATURE_VECTOR_DIM= 16
DROPOUT_RATE      = 0.3
RF_N_ESTIMATORS   = 200
RF_MAX_DEPTH      = None
SMOTE_SAMPLING    = "auto"
PATIENCE          = 5

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

from sklearn.preprocessing       import LabelEncoder, StandardScaler
from sklearn.impute               import SimpleImputer
from sklearn.feature_selection   import SelectKBest, f_classif
from sklearn.ensemble            import RandomForestClassifier
from sklearn.metrics             import (classification_report,
                                          confusion_matrix,
                                          precision_recall_curve,
                                          average_precision_score,
                                          accuracy_score,
                                          precision_score, recall_score,
                                          f1_score)
from imblearn.over_sampling      import SMOTE
import joblib

import tensorflow as tf
from tensorflow.keras.models     import Model
from tensorflow.keras.layers     import (Input, LSTM, Dense,
                                          Dropout, BatchNormalization)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_CSV = "UNSW_NB15_training-set.csv"
TEST_CSV  = "UNSW_NB15_testing-set.csv"
IMG_DIR   = "images_H2"
MODEL_DIR = "models_H2"
os.makedirs(IMG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
#  1. LOAD DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 65)
print("  MODEL H2 : LSTM + Random Forest  –  UNSW-NB15")
print("=" * 65)

df_train = pd.read_csv(TRAIN_CSV)
df_test  = pd.read_csv(TEST_CSV)
print(f"Train shape : {df_train.shape}   |   Test shape : {df_test.shape}")

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
    cat_te = df_te["attack_cat"].values if "attack_cat" in df_te.columns else None

    feat_cols = [c for c in df_tr.columns if c not in ("label", "attack_cat")]
    X_tr = df_tr[feat_cols].values.astype(np.float32)
    X_te = df_te[feat_cols].values.astype(np.float32)

    imp  = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(X_tr)
    X_te = imp.transform(X_te)

    sel  = SelectKBest(f_classif, k=min(K_BEST_FEATURES, X_tr.shape[1]))
    X_tr = sel.fit_transform(X_tr, y_tr)
    X_te = sel.transform(X_te)
    selected_features = [feat_cols[i] for i in sel.get_support(indices=True)]
    print(f"[Feature Selection] {len(selected_features)} features selected: {selected_features}")

    sc   = StandardScaler()
    X_tr = sc.fit_transform(X_tr)
    X_te = sc.transform(X_te)

    return X_tr, X_te, y_tr, y_te, cat_tr, cat_te, selected_features


X_train, X_test, y_train, y_test, cat_train, cat_test, feat_names = \
    preprocess(df_train.copy(), df_test.copy())

# ─────────────────────────────────────────────────────────────────────────────
#  3. VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────
plt.figure(figsize=(6, 4))
unique, counts = np.unique(y_train, return_counts=True)
plt.bar([str(u) for u in unique], counts, color=["steelblue","tomato"])
plt.title("Class Distribution (Train) – H2")
plt.xlabel("Label (0=Normal, 1=Attack)"); plt.ylabel("Count")
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/01_class_distribution.png", dpi=150); plt.close()

if cat_train is not None:
    plt.figure(figsize=(10, 5))
    pd.Series(cat_train).value_counts().plot(
        kind="bar", color="steelblue", edgecolor="k")
    plt.title("Attack Type Distribution (Train) – H2")
    plt.ylabel("Count"); plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{IMG_DIR}/02_attack_types.png", dpi=150); plt.close()

print("[Plot] Class & attack distribution saved.")

# ─────────────────────────────────────────────────────────────────────────────
#  4. SMOTE
# ─────────────────────────────────────────────────────────────────────────────
print("[SMOTE] Balancing …")
sm = SMOTE(sampling_strategy=SMOTE_SAMPLING, random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
print(f"[SMOTE] After: {X_train_sm.shape} | {dict(zip(*np.unique(y_train_sm, return_counts=True)))}")

# ─────────────────────────────────────────────────────────────────────────────
#  5. RESHAPE
# ─────────────────────────────────────────────────────────────────────────────
X_train_3d = X_train_sm.reshape(-1, 1, X_train_sm.shape[1])
X_test_3d  = X_test.reshape(-1, 1, X_test.shape[1])

# ─────────────────────────────────────────────────────────────────────────────
#  6. LSTM FEATURE EXTRACTOR  (uni-directional)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[LSTM] Building model …")
n_features = X_train_3d.shape[2]
inp = Input(shape=(1, n_features), name="input")
x   = LSTM(LSTM_UNITS, return_sequences=True, name="lstm_1")(inp)
x   = Dropout(DROPOUT_RATE)(x)
x   = LSTM(LSTM_UNITS // 2, return_sequences=False, name="lstm_2")(x)
x   = BatchNormalization()(x)
x   = Dropout(DROPOUT_RATE)(x)
feat_out = Dense(FEATURE_VECTOR_DIM, activation="relu", name="feature_vector")(x)
clf_out  = Dense(1, activation="sigmoid", name="classifier")(feat_out)

lstm_full = Model(inputs=inp, outputs=clf_out, name="LSTM_classifier")
lstm_full.compile(optimizer=Adam(LEARNING_RATE),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
lstm_full.summary()

callbacks = [
    EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(factor=0.5, patience=3, verbose=1)
]

print("\n[LSTM] Training …")
t0 = time.time()
history = lstm_full.fit(
    X_train_3d, y_train_sm,
    epochs=EPOCHS, batch_size=BATCH_SIZE,
    validation_split=TEST_SIZE,
    callbacks=callbacks, verbose=1
)
lstm_train_time = time.time() - t0
print(f"[LSTM] Done in {lstm_train_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
#  7. LOSS CURVE
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history["loss"],     label="Train")
axes[0].plot(history.history["val_loss"], label="Val")
axes[0].set_title("Loss Curve – H2"); axes[0].legend()
axes[1].plot(history.history["accuracy"],     label="Train")
axes[1].plot(history.history["val_accuracy"], label="Val")
axes[1].set_title("Accuracy Curve – H2"); axes[1].legend()
plt.tight_layout()
plt.savefig(f"{IMG_DIR}/03_lstm_loss_accuracy.png", dpi=150); plt.close()
print("[Plot] Loss/Accuracy curves saved.")

# ─────────────────────────────────────────────────────────────────────────────
#  8. EXTRACT FEATURE VECTORS
# ─────────────────────────────────────────────────────────────────────────────
feature_extractor = Model(inputs=inp,
                          outputs=lstm_full.get_layer("feature_vector").output,
                          name="FeatureExtractor")
print("\n[LSTM] Extracting feature vectors …")
X_train_feat = feature_extractor.predict(X_train_3d, batch_size=BATCH_SIZE, verbose=0)
X_test_feat  = feature_extractor.predict(X_test_3d,  batch_size=BATCH_SIZE, verbose=0)

# ─────────────────────────────────────────────────────────────────────────────
#  9. RANDOM FOREST
# ─────────────────────────────────────────────────────────────────────────────
print("\n[RF] Training …")
t0 = time.time()
rf = RandomForestClassifier(
    n_estimators=RF_N_ESTIMATORS,
    max_depth=RF_MAX_DEPTH,
    class_weight="balanced",
    random_state=RANDOM_STATE, n_jobs=-1
)
rf.fit(X_train_feat, y_train_sm)
rf_train_time = time.time() - t0
total_time    = lstm_train_time + rf_train_time

y_pred  = rf.predict(X_test_feat)
y_proba = rf.predict_proba(X_test_feat)[:, 1]
acc     = accuracy_score(y_test, y_pred)
report  = classification_report(y_test, y_pred,
                                 target_names=["Normal","Attack"], digits=4)
print("\n" + "=" * 65)
print("  CLASSIFICATION REPORT – Model H2 (LSTM + RF)")
print("=" * 65)
print(report)
print(f"  Accuracy: {acc:.4f}  |  Time: {total_time:.1f}s")

# ─────────────────────────────────────────────────────────────────────────────
#  10. CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal","Attack"], yticklabels=["Normal","Attack"])
plt.title("Confusion Matrix – H2 (LSTM + RF)")
plt.xlabel("Predicted"); plt.ylabel("True")
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/04_confusion_matrix.png", dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  11. FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────────────────
feat_imp   = rf.feature_importances_
feat_labels= [f"LSTM_feat_{i}" for i in range(FEATURE_VECTOR_DIM)]
sorted_idx = np.argsort(feat_imp)[::-1]
plt.figure(figsize=(8, 5))
plt.bar(range(FEATURE_VECTOR_DIM), feat_imp[sorted_idx],
        color="steelblue", edgecolor="k")
plt.xticks(range(FEATURE_VECTOR_DIM), [feat_labels[i] for i in sorted_idx], rotation=90)
plt.title("Feature Importance – H2 (RF on LSTM features)")
plt.tight_layout(); plt.savefig(f"{IMG_DIR}/05_feature_importance.png", dpi=150); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
#  12. PRECISION-RECALL CURVE
# ─────────────────────────────────────────────────────────────────────────────
prec, rec, _ = precision_recall_curve(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
plt.figure(figsize=(7, 5))
plt.plot(rec, prec, color="darkorange", lw=2, label=f"AP = {ap:.4f}")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall Curve – H2 (LSTM + RF)")
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

result_text = f"""
============================================================
  RESULTS – Model H2 : LSTM + Random Forest (UNSW-NB15)
============================================================
Accuracy       : {acc:.4f}
Precision (W)  : {prec_w:.4f}
Recall    (W)  : {rec_w:.4f}
F1-Score  (W)  : {f1_w:.4f}
Training time  : {total_time:.1f} s
Average Prec.  : {ap:.4f}

-- Class: With Anomaly (Attack=1) --
Precision : {prec_a:.4f}
Recall    : {rec_a:.4f}
F1-Score  : {f1_a:.4f}

-- Class: Without Anomaly (Normal=0) --
Precision : {prec_n:.4f}
Recall    : {rec_n:.4f}
F1-Score  : {f1_n:.4f}

------------------------------------------------------------
Selected Features (SelectKBest top {K_BEST_FEATURES}):
{feat_names}
------------------------------------------------------------
LSTM extracted feature dimension : {FEATURE_VECTOR_DIM}
SMOTE applied                    : Yes
"""
with open(f"{IMG_DIR}/results_H2.txt", "w") as f:
    f.write(result_text)
print("[Results] Saved."); print(result_text)

# ─────────────────────────────────────────────────────────────────────────────
#  14. SAVE MODELS
# ─────────────────────────────────────────────────────────────────────────────
lstm_full.save(f"{MODEL_DIR}/lstm_model_H2.h5")
feature_extractor.save(f"{MODEL_DIR}/feature_extractor_H2.h5")
joblib.dump(rf, f"{MODEL_DIR}/random_forest_H2.joblib")
print(f"[Save] Models saved in '{MODEL_DIR}/'\nDone – Model H2.")
