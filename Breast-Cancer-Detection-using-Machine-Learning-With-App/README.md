# Breast Cancer Risk Assessment

A machine learning web application that predicts whether a breast tumor sample is likely **benign** or **malignant** from numeric tumor measurements.

This project combines data analysis, model training, and a Flask web interface. The model is trained on breast cancer diagnosis data and served through a user-friendly prediction page where users can paste the required feature values and receive a result.

> Important: This application is an educational machine learning project. It is not a medical diagnosis tool and should not replace advice from a qualified healthcare professional.

---

## Project Highlights

- Breast cancer classification using machine learning
- Cleaned and processed dataset with tumor measurement features
- Logistic Regression model saved as `model/base_model.pkl`
- Flask web app for browser-based predictions
- Modern UI with input validation and feature guidance
- Sample input button for quick testing
- Prediction confidence shown when available

---

## Dataset Overview

The dataset contains breast cancer diagnosis records where each row represents one patient/sample.

| Item | Details |
| --- | --- |
| Rows | 569 |
| Original columns | 33 |
| Target column | `diagnosis` |
| Benign cases | 357 |
| Malignant cases | 212 |
| Dropped column | `Unnamed: 32` |
| Model input features | 31 |

The target variable is:

| Label | Meaning | Encoded value |
| --- | --- | --- |
| `B` | Benign | `0` |
| `M` | Malignant | `1` |

The features describe tumor characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

---

## Machine Learning Workflow

The notebook follows this general workflow:

1. Load the breast cancer dataset.
2. Explore the data using summary statistics and visualizations.
3. Drop the empty `Unnamed: 32` column.
4. Encode the diagnosis column:

```python
br["diagnosis"] = br["diagnosis"].map({"M": 1, "B": 0})
```

5. Split the data into training and testing sets.
6. Scale the features with `StandardScaler`.
7. Train a Logistic Regression model.
8. Export the trained model as:

```text
model/base_model.pkl
```

Current notebook-style model accuracy:

```text
Accuracy: 0.967
```

---

## Project Structure

```text
Breast-Cancer-Detection-using-Machine-Learning-With-App/
|-- app.py
|-- pyproject.toml
|-- README.md
|-- model/
|   `-- base_model.pkl
|-- notebook/
|   |-- cancer_prediction.ipynb
|   `-- Breast Cancer Project Analysis Report.docx
|-- raw_data/
|   `-- breast cancer.csv
|-- static/
|   |-- alert_imge.png
|   |-- img.jpg
|   `-- okay_img.jpg
`-- templates/
    `-- index.html
```

---

## How To Run The Web App

### 1. Open PowerShell

Go to the main project workspace:

```powershell
cd "C:\Users\Sayed Al Rafi\Documents\Data Desktop\Data_Analysis\ML project\Breast Cancer Prediction(PJ 1)"
```

### 2. Run the Flask app

Use the existing virtual environment:

```powershell
.venv\Scripts\python.exe Breast-Cancer-Detection-using-Machine-Learning-With-App\app.py
```

### 3. Open the app in your browser

```text
http://127.0.0.1:5000
```

### 4. Stop the app

Press this in the terminal:

```text
Ctrl + C
```

---

## How To Install Dependencies

If you are setting up the project on a new machine, create and activate a virtual environment first.

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Install the project dependencies from `pyproject.toml`:

```powershell
pip install -e Breast-Cancer-Detection-using-Machine-Learning-With-App
```

Required Python version:

```text
Python 3.11 or newer
```

---

## How To Use The Prediction Page

The web app expects **31 numeric values** in the same order used during model training.

You can enter the values separated by:

- commas
- spaces
- new lines

Example input:

```text
-0.22113181, 0.82993333, 1.41380849, 0.87579342, 0.75772485,
0.7147203, 1.36344147, 0.81911538, 0.68080163, 0.32352939,
0.2485258, 0.58266418, -0.32516484, 0.33546888, 0.63173316,
-0.49934293, 1.56191362, 0.16132205, -0.16850851, -0.59779178,
0.6694212, 1.30228984, 1.47946329, 1.16948048, 1.2100149,
1.14282204, 2.96420707, 1.75078692, 1.06666888, 0.5676593,
2.75263383
```

The app validates the number of values before making a prediction. If the input is wrong, it shows a helpful error message.

---

## Feature Input Order

Keep this exact order when entering values:

```text
1. id
2. radius_mean
3. texture_mean
4. perimeter_mean
5. area_mean
6. smoothness_mean
7. compactness_mean
8. concavity_mean
9. concave points_mean
10. symmetry_mean
11. fractal_dimension_mean
12. radius_se
13. texture_se
14. perimeter_se
15. area_se
16. smoothness_se
17. compactness_se
18. concavity_se
19. concave points_se
20. symmetry_se
21. fractal_dimension_se
22. radius_worst
23. texture_worst
24. perimeter_worst
25. area_worst
26. smoothness_worst
27. compactness_worst
28. concavity_worst
29. concave points_worst
30. symmetry_worst
31. fractal_dimension_worst
```

Note: The current model includes the `id` column because it was present during notebook training. In a future production version, the ID should usually be removed from model features.

---

## Using The Model In Python

You can load the saved model directly:

```python
import pickle
import numpy as np

model_path = "model/base_model.pkl"

with open(model_path, "rb") as file:
    model = pickle.load(file)

sample = np.array([
    -0.22113181, 0.82993333, 1.41380849, 0.87579342, 0.75772485,
    0.7147203, 1.36344147, 0.81911538, 0.68080163, 0.32352939,
    0.2485258, 0.58266418, -0.32516484, 0.33546888, 0.63173316,
    -0.49934293, 1.56191362, 0.16132205, -0.16850851, -0.59779178,
    0.6694212, 1.30228984, 1.47946329, 1.16948048, 1.2100149,
    1.14282204, 2.96420707, 1.75078692, 1.06666888, 0.5676593,
    2.75263383
])

prediction = model.predict(sample.reshape(1, -1))

if prediction[0] == 1:
    print("Possible malignant tumor")
else:
    print("Likely benign tumor")
```

---

## Main Files

| File | Purpose |
| --- | --- |
| `app.py` | Flask application, model loading, input validation, prediction route |
| `templates/index.html` | Web page UI for user input and prediction result |
| `model/base_model.pkl` | Saved trained model |
| `notebook/cancer_prediction.ipynb` | Data analysis, preprocessing, training, and model export |
| `raw_data/breast cancer.csv` | Original dataset |
| `pyproject.toml` | Project metadata and dependencies |

---

## Current Limitations

- The model currently expects standardized feature values.
- The saved pickle contains the model only, not the scaler.
- The `id` column is currently included as an input feature.
- Pickle files should only be loaded from trusted sources.
- The app is for educational use and should not be used as a medical decision system.

Recommended future improvement:

```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ("scaler", scaler),
    ("model", lr),
])
```

Saving a full pipeline would let the app accept raw feature values and handle scaling automatically.

---

## Future Improvements

- Save and load a full preprocessing pipeline
- Remove `id` from model training features
- Add individual input fields for each tumor feature
- Add charts for prediction confidence
- Add model comparison metrics such as precision, recall, F1 score, and confusion matrix
- Add deployment support for Render, Railway, or Docker

---

## Tech Stack

- Python
- Flask
- NumPy
- Pandas
- Scikit-learn
- HTML
- CSS
- Bootstrap

---

## License

This project is intended for learning and portfolio use. Add a license file if you plan to publish or distribute it publicly.
