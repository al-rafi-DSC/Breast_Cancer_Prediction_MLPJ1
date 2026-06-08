from pathlib import Path
import pickle
import re

import numpy as np
from flask import Flask, render_template, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model" / "base_model.pkl"

FEATURE_NAMES = [
    "id",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]

SAMPLE_INPUT = (
    "-0.22113181, 0.82993333, 1.41380849, 0.87579342, 0.75772485, "
    "0.7147203, 1.36344147, 0.81911538, 0.68080163, 0.32352939, "
    "0.2485258, 0.58266418, -0.32516484, 0.33546888, 0.63173316, "
    "-0.49934293, 1.56191362, 0.16132205, -0.16850851, -0.59779178, "
    "0.6694212, 1.30228984, 1.47946329, 1.16948048, 1.2100149, "
    "1.14282204, 2.96420707, 1.75078692, 1.06666888, 0.5676593, "
    "2.75263383"
)

app = Flask(__name__)

with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)


def parse_features(raw_features):
    values = [item for item in re.split(r"[\s,]+", raw_features.strip()) if item]

    if not values:
        raise ValueError("Enter the 31 standardized feature values before predicting.")

    if len(values) != len(FEATURE_NAMES):
        raise ValueError(
            f"Expected {len(FEATURE_NAMES)} values, but received {len(values)}."
        )

    try:
        return np.asarray(values, dtype=np.float32).reshape(1, -1)
    except ValueError as exc:
        raise ValueError("Only numeric values are accepted.") from exc


def render_home(**context):
    page_context = {
        "feature_names": FEATURE_NAMES,
        "sample_input": SAMPLE_INPUT,
        "expected_count": len(FEATURE_NAMES),
    }
    page_context.update(context)
    return render_template("index.html", **page_context)


@app.route("/")
def home():
    return render_home()


@app.route("/predict", methods=["POST"])
def predict():
    input_value = request.form.get("feature", "")

    try:
        np_features = parse_features(input_value)
        prediction = int(model.predict(np_features)[0])
        probability = None

        if hasattr(model, "predict_proba"):
            probability = round(float(model.predict_proba(np_features)[0][prediction]) * 100, 1)

        result = {
            "label": "Possible malignant tumor" if prediction == 1 else "Likely benign tumor",
            "status": "warning" if prediction == 1 else "success",
            "description": (
                "The model classified this sample as malignant. Please treat this as a screening output, not a medical diagnosis."
                if prediction == 1
                else "The model classified this sample as benign. This is still only a machine-learning estimate."
            ),
            "probability": probability,
        }
        return render_home(result=result, input_value=input_value)
    except ValueError as exc:
        return render_home(error=str(exc), input_value=input_value)


if __name__ == "__main__":
    app.run(debug=True)
