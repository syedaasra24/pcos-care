# ==========================================================
# PCOS CARE
# AI MODEL TRAINING & COMPARISON
# Random Forest + XGBoost
# ==========================================================

import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from xgboost import XGBClassifier

# ==========================================================
# LOAD DATASET
# ==========================================================

print("=" * 60)
print("PCOS CARE - MODEL TRAINING")
print("=" * 60)

try:

    df = pd.read_csv("data/pcos_balanced.csv")

    print("✅ Dataset Loaded Successfully")
    print(f"Dataset Shape : {df.shape}")

except FileNotFoundError:

    print("❌ Dataset not found!")
    exit()

# ==========================================================
# REMOVE DUPLICATES
# ==========================================================

duplicates = df.duplicated().sum()

if duplicates > 0:

    print(f"⚠ Removing {duplicates} duplicate rows...")

    df.drop_duplicates(inplace=True)

print(f"Current Shape : {df.shape}")

# ==========================================================
# FEATURES & TARGET
# ==========================================================

X = df.drop("PCOS", axis=1)
y = df["PCOS"]

categorical_features = [

    "Country",
    "Cycle_Regularity",
    "Hyperandrogenism",
    "Acne",
    "Hair_Thinning",
    "Skin_Darkening",
    "Rapid_Weight_Gain",
    "Stress_Level",
    "Lifestyle_Score"

]

numeric_features = [

    "Age",
    "BMI",
    "Cycle_Length",
    "Hirsutism_Score"

]

# ==========================================================
# PREPROCESSING
# ==========================================================

numeric_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(

    transformers=[

        ("num", numeric_transformer, numeric_features),

        ("cat", categorical_transformer, categorical_features)

    ]

)

# ==========================================================
# RANDOM FOREST PIPELINE
# ==========================================================

rf_pipeline = Pipeline(

    steps=[

        ("preprocessor", preprocessor),

        ("classifier",

            RandomForestClassifier(

                n_estimators=100,
                max_depth=10,
                random_state=42

            )

        )

    ]

)

# ==========================================================
# XGBOOST PIPELINE
# ==========================================================

xgb_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier",
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=5,
                min_child_weight=5,
                gamma=0.1,
                subsample=0.9,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=42,
                eval_metric="logloss"
            )
        )
    ]
)

# ==========================================================
# TRAIN TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y

)

print("-" * 60)

print(f"Training Samples : {len(X_train)}")

print(f"Testing Samples  : {len(X_test)}")

print("-" * 60)

# ==========================================================
# CROSS-VALIDATION (more reliable than a single split)
# ==========================================================

print("\n📊 Running 5-fold cross-validation on the training set...")

cv_splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scoring = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}

cv_results = {}

for name, pipe in [("Random Forest", rf_pipeline), ("XGBoost", xgb_pipeline)]:

    scores = cross_validate(pipe, X_train, y_train, cv=cv_splitter, scoring=cv_scoring)

    cv_results[name] = {
        metric: (float(np.mean(scores[f"test_{metric}"])), float(np.std(scores[f"test_{metric}"])))
        for metric in cv_scoring
    }

    r = cv_results[name]
    print(f"{name:14s} CV  Acc {r['accuracy'][0]*100:.2f}% (+/- {r['accuracy'][1]*100:.2f})  "
          f"Recall {r['recall'][0]*100:.2f}%  F1 {r['f1'][0]*100:.2f}%  AUC {r['roc_auc'][0]:.3f}")

# ==========================================================
# TRAIN RANDOM FOREST
# ==========================================================

print("\n🌲 Training Random Forest...")

rf_start = time.time()

rf_pipeline.fit(X_train, y_train)

rf_training_time = time.time() - rf_start

rf_predictions = rf_pipeline.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)

print("✅ Random Forest Training Completed!")

print("\nRandom Forest Classification Report")
print(classification_report(y_test, rf_predictions))

print("\nRandom Forest Confusion Matrix")
print(confusion_matrix(y_test, rf_predictions))


# ==========================================================
# TRAIN XGBOOST
# ==========================================================

print("\n🚀 Training XGBoost...")

xgb_start = time.time()

xgb_pipeline.fit(X_train, y_train)

xgb_training_time = time.time() - xgb_start

xgb_predictions = xgb_pipeline.predict(X_test)

xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)

print("✅ XGBoost Training Completed!")

print("\nXGBoost Classification Report")
print(classification_report(y_test, xgb_predictions))

print("\nXGBoost Confusion Matrix")
print(confusion_matrix(y_test, xgb_predictions))


# ==========================================================
# MODEL COMPARISON
# ==========================================================

print("\n")
print("=" * 70)
print("                 PCOS CARE MODEL COMPARISON")
print("=" * 70)

print("\n🌲 RANDOM FOREST")
print(f"Accuracy      : {rf_accuracy * 100:.2f}%")
print(f"Precision     : {rf_precision * 100:.2f}%")
print(f"Recall        : {rf_recall * 100:.2f}%")
print(f"F1 Score      : {rf_f1 * 100:.2f}%")
print(f"Training Time : {rf_training_time:.3f} seconds")

print("\n🚀 XGBOOST")
print(f"Accuracy      : {xgb_accuracy * 100:.2f}%")
print(f"Precision     : {xgb_precision * 100:.2f}%")
print(f"Recall        : {xgb_recall * 100:.2f}%")
print(f"F1 Score      : {xgb_f1 * 100:.2f}%")
print(f"Training Time : {xgb_training_time:.3f} seconds")

# For a screening tool, missing a real PCOS case is worse than a false
# alarm, so pick the model using F2 (recall counts double) from cross-validation.
def cv_f2(r):
    p, rc = r["precision"][0], r["recall"][0]
    return 5 * p * rc / (4 * p + rc)

if cv_f2(cv_results["XGBoost"]) >= cv_f2(cv_results["Random Forest"]):
    best_model = "XGBoost"
    best_pipeline = xgb_pipeline
else:
    best_model = "Random Forest"
    best_pipeline = rf_pipeline

print("\n🏆 BEST MODEL :", best_model, "(chosen by cross-validated F2)")

print("=" * 70)

# ==========================================================
# SAVE MODELS
# ==========================================================

def test_metrics(pipe, preds):
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds)),
        "recall": float(recall_score(y_test, preds)),
        "f1": float(f1_score(y_test, preds)),
        "f2": float(fbeta_score(y_test, preds, beta=2)),
        "roc_auc": float(roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])),
    }

metrics = {
    "selected_model": best_model,
    "selection_rule": "highest cross-validated F2 (recall-weighted)",
    "train_samples": int(len(X_train)),
    "test_samples": int(len(X_test)),
    "random_forest": {
        "test": test_metrics(rf_pipeline, rf_predictions),
        "cv_mean_std": cv_results["Random Forest"],
        "training_time_sec": rf_training_time,
    },
    "xgboost": {
        "test": test_metrics(xgb_pipeline, xgb_predictions),
        "cv_mean_std": cv_results["XGBoost"],
        "training_time_sec": xgb_training_time,
    },
}

with open("model_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("📄 Saved model_metrics.json")

joblib.dump(rf_pipeline, "random_forest.joblib")
joblib.dump(xgb_pipeline, "xgboost.joblib")

# Save the best performing model for the Flask app
joblib.dump(best_pipeline, "pcos_pipeline.joblib")

print("\n💾 Models Saved Successfully!")
print("✔ random_forest.joblib")
print("✔ xgboost.joblib")
print("✔ pcos_pipeline.joblib (Best Model)")


# ==========================================================
# MANUAL TEST
# ==========================================================

print("\n🧪 Running Manual Test...")

sample_patient = pd.DataFrame({

    'Country': ['India'],
    'Age': [26],
    'BMI': [22.0],
    'Cycle_Length': [28],
    'Cycle_Regularity': ['Regular'],
    'Hirsutism_Score': [2],
    'Hyperandrogenism': ['No'],
    'Acne': ['No'],
    'Hair_Thinning': ['No'],
    'Skin_Darkening': ['No'],
    'Rapid_Weight_Gain': ['No'],
    'Stress_Level': ['Low'],
    'Lifestyle_Score': ['Moderate']

})

rf_prediction = rf_pipeline.predict(sample_patient)[0]
xgb_prediction = xgb_pipeline.predict(sample_patient)[0]
best_prediction = best_pipeline.predict(sample_patient)[0]

print("\n" + "=" * 70)
print("                 MANUAL TEST RESULTS")
print("=" * 70)

print(f"🌲 Random Forest : {'High Risk' if rf_prediction == 1 else 'Low Risk'}")
print(f"🚀 XGBoost       : {'High Risk' if xgb_prediction == 1 else 'Low Risk'}")
print(f"🏆 Best Model    : {'High Risk' if best_prediction == 1 else 'Low Risk'}")

print("=" * 70)

print("\n🎉 TRAINING COMPLETED SUCCESSFULLY!")

print(f"""
==========================================================
PCOS CARE AI TRAINING SUMMARY
==========================================================

🌲 Random Forest Accuracy : {rf_accuracy * 100:.2f}%

🚀 XGBoost Accuracy        : {xgb_accuracy * 100:.2f}%

🏆 Selected Model          : {best_model}

Saved Models:

✔ random_forest.joblib

✔ xgboost.joblib

✔ pcos_pipeline.joblib

==========================================================
""")