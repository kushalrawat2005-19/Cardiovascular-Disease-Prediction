# Cardiovascular Disease Prediction

Machine-learning project that predicts heart-disease status from health and lifestyle data — **308,854 records, 18 features** — comparing SVM, Random Forest and Logistic Regression, with hyperparameter tuning and an honest look at class imbalance.

> Major Project — Data Science Internship at **Elewayte** (Internship Partner: Cubestech)

![Python](https://img.shields.io/badge/Python-3.x-blue) ![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange) ![pandas](https://img.shields.io/badge/pandas-Data%20Analysis-orange) ![Seaborn](https://img.shields.io/badge/Seaborn-Charts-teal) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Dataset

- **Size:** 308,854 records × 19 columns (18 features + target)
- **Target:** `Heart_Disease` (Yes / No)
- **Features include:** general health, checkup history, exercise, skin/other cancer, depression, diabetes, arthritis, sex, age category, height, weight, BMI, smoking history, alcohol / fruit / green-vegetable / fried-potato consumption
- **Note:** the notebook expects the dataset as `CVD_cleaned.csv` in the notebook folder (upload prompt is built into the first cells). The dataset file itself is not included in this repo.

## Approach

1. **Cleaning pipeline** — missing-value checks, encoding of categorical features, feature scaling
2. **Split** — 80:20 stratified train–test split (test set: 61,771 records)
3. **Models compared** — Support Vector Machine, Random Forest, Logistic Regression
4. **Tuning** — `RandomizedSearchCV` for each model
5. **Evaluation** — accuracy, precision, recall, F1-score and confusion matrices

## Results

| Model | Accuracy (before tuning) |
|---|---|
| SVM | **91.92%** |
| Random Forest | 91.83% |
| Logistic Regression | 73.83% |

### The key insight — accuracy lies on imbalanced data

Only **~8% of cases are positive** (4,994 of 61,771 in the test set). The high-accuracy models still miss most actual heart-disease cases, so the models were judged on **recall / F1 for the positive class**, not accuracy alone:

- **Logistic Regression caught 80% of positive cases** (recall 0.80 for the "Yes" class) — the best positive-case detection of the three, despite the lowest headline accuracy.

## Files

| File | Description |
|---|---|
| `Cardiovascular_Disease_Prediction.ipynb` | Main notebook with full pipeline, results and charts |
| `requirements.txt` | Python dependencies |

## How to Run

```bash
pip install -r requirements.txt
# place CVD_cleaned.csv next to the notebook
jupyter notebook Cardiovascular_Disease_Prediction.ipynb
```

## Tools Used

Python · scikit-learn · pandas · NumPy · Matplotlib · Seaborn · Jupyter / Google Colab

## Disclaimer

This is an academic machine-learning project. Its predictions are **not a medical diagnosis**.

## Author

**Kushal Rawat** — Data Analyst | BCA (AI & Data Science), JECRC University, Jaipur
[LinkedIn](https://www.linkedin.com/in/kushal-rawat-data/) · [GitHub](https://github.com/kushalrawat2005-19)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

