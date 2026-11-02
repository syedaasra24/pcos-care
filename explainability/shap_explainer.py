import shap
import joblib

# Explain the SAME model the app uses for its final prediction
model = joblib.load("pcos_pipeline.joblib")

explainer = shap.Explainer(model.named_steps["classifier"])

FEATURE_LABELS = {
    "Age": "Age",
    "BMI": "BMI",
    "Cycle_Length": "Cycle length",
    "Hirsutism_Score": "Excess hair growth",
    "Cycle_Regularity": "Cycle regularity",
    "Hyperandrogenism": "High androgen signs",
    "Acne": "Acne",
    "Hair_Thinning": "Hair thinning",
    "Skin_Darkening": "Skin darkening",
    "Rapid_Weight_Gain": "Rapid weight gain",
    "Stress_Level": "Stress level",
    "Lifestyle_Score": "Lifestyle",
    "Country": "Country",
}


def _readable(name):
    """'cat__Acne_Yes' -> 'Acne: Yes', 'num__BMI' -> 'BMI'"""
    name = name.split("__", 1)[-1]
    for key, label in FEATURE_LABELS.items():
        if name == key:
            return label
        if name.startswith(key + "_"):
            return f"{label}: {name[len(key) + 1:]}"
    return name


def explain_prediction(df):

    transformed = model.named_steps["preprocessor"].transform(df)

    shap_values = explainer(transformed)

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()

    values = shap_values.values[0]

    # Random Forest returns one column per class; keep the PCOS class (1)
    if values.ndim == 2:
        values = values[:, 1]

    row = transformed[0]
    row = row.toarray().ravel() if hasattr(row, "toarray") else row

    features = []

    for name, value, x in zip(feature_names, values, row):

        # Skip one-hot options the person did NOT choose (e.g. "Regular"
        # when they answered "Irregular"); only explain their own answers.
        if name.startswith("cat__") and x == 0:
            continue

        features.append({
            "feature": _readable(name),
            "importance": abs(float(value)),
            "direction": "raises" if value > 0 else "lowers"
        })

    features = sorted(
        features,
        key=lambda x: x["importance"],
        reverse=True
    )

    return features[:5]