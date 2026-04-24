# Samay Shetty
## Glowlogics Project 4

---

# Disease Prediction System using Machine Learning
### Healthcare Analytics

---


## Project Overview

This project builds an intelligent Disease Prediction System that uses Machine Learning to predict diseases based on patient symptoms and clinical data. It is designed to support doctors in early diagnosis, reduce manual analysis time, and improve overall healthcare decision-making.

The system is trained on 2,000 patient records covering 8 disease categories. It compares five classification algorithms and selects the best-performing model to make accurate predictions along with a risk level assessment.

---

## Objectives

- Predict diseases from patient symptoms and medical parameters
- Assist healthcare professionals with data-driven early diagnosis
- Provide risk level classification alongside disease prediction
- Reduce diagnostic time and cost through automation
- Compare multiple ML models and identify the best performer

---

## Problem Statement

In traditional healthcare:

- Diagnosis depends heavily on doctor experience, which can vary
- Early detection of diseases like Diabetes or Heart Disease is often missed
- Manually analyzing symptoms across multiple patients is time-consuming
- Ordering unnecessary tests increases both time and cost for patients

This project addresses these challenges by automating the diagnostic process using machine learning trained on real patient data.

---

## Dataset

| Property | Details |
|----------|---------|
| Total Records | 2,000 patients |
| Total Features | 20 raw features |
| Engineered Features | 3 additional (Symptom Count, BP Pulse Pressure, Risk Score) |
| Target Variable | Disease (8 classes) |
| Missing Data | Medical History column (~62% missing, handled by imputation) |

### Disease Classes

| Disease | Description |
|---------|-------------|
| Flu | Common viral infection |
| Diabetes | High blood sugar condition |
| Healthy | No disease detected |
| Hypertension | High blood pressure |
| Heart Disease | Cardiovascular condition |
| Pneumonia | Lung infection |
| Asthma | Respiratory condition |
| Migraine | Chronic headache disorder |

### Feature Categories

**Demographic** — Age, Gender

**Binary Symptoms (0 = Absent, 1 = Present)** — Fever, Cough, Fatigue, Body Ache, Headache, Chest Pain, Shortness of Breath, Nausea, Dizziness, Frequent Urination, Increased Thirst, Blurred Vision

**Clinical Parameters** — Blood Pressure, Sugar Level, Cholesterol, Medical History

**Engineered Features:**
- `Symptom_Count` — Total number of active symptoms out of 12
- `BP_Pulse_Pressure` — Difference between Systolic and Diastolic BP
- `Risk_Score` — Weighted composite of symptoms and abnormal clinical values

---

## Technologies Used

| Tool / Library | Purpose |
|---------------|---------|
| Python 3.11+ | Core programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical computations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Scikit-Learn | Machine learning models and evaluation |
| Jupyter Notebook | Development environment |

---

## Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/samayshetty/disease-prediction.git
cd disease-prediction
```

**Step 2 — Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

**Step 3 — Install required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**Step 4 — Launch the notebook**

```bash
jupyter notebook Disease_Prediction_ML.ipynb
```

**Step 5 — Run all cells**

Go to Kernel → Restart & Run All

> Make sure `dataset.csv` is placed in the same folder as the notebook before running.

---

## How It Works

The system follows a structured 8-step pipeline:

**Step 1 — Data Collection**
Load the patient dataset from the CSV file containing symptoms, clinical values, and disease labels.

**Step 2 — Data Preprocessing**
- Remove the non-informative Patient ID column
- Split the Blood Pressure column into Systolic and Diastolic values
- Encode categorical columns (Gender, Medical History, Disease) using Label Encoding
- Fill missing Medical History values with "None"

**Step 3 — Exploratory Data Analysis**
Visualize the data to understand patterns — disease distribution, age ranges per disease, symptom prevalence, clinical metric distributions, gender breakdown, and feature correlations.

**Step 4 — Feature Engineering**
Create three new features to strengthen model performance — Symptom Count, BP Pulse Pressure, and a custom Risk Score based on clinical thresholds.

**Step 5 — Model Selection**
Apply five different classification algorithms to the dataset and compare their performance.

**Step 6 — Model Training**
Split data into 80% training and 20% testing. Train all models on the training set. Use 5-Fold Cross-Validation to assess generalization.

**Step 7 — Prediction**
Feed patient data into the trained model to receive a predicted disease, risk level, and top-3 probability scores for all possible diseases.

**Step 8 — Evaluation**
Measure each model using Accuracy, Precision, Recall, F1-Score, and Cross-Validation scores.

---

## Machine Learning Models

| Model | Type | Notes |
|-------|------|-------|
| Logistic Regression | Linear classifier | Scaled data, max 1000 iterations |
| Decision Tree | Tree-based | Max depth 10 |
| Random Forest | Ensemble (trees) | 200 estimators, max depth 15 |
| Naive Bayes | Probabilistic | Gaussian distribution assumed |
| SVM | Kernel-based | RBF kernel, scaled data |

---

## Results

### Model Performance Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Random Forest ⭐ | ~92% | ~92% | ~92% | ~92% |
| Decision Tree | ~88% | ~88% | ~88% | ~88% |
| SVM | ~85% | ~85% | ~85% | ~85% |
| Logistic Regression | ~80% | ~80% | ~80% | ~80% |
| Naive Bayes | ~74% | ~74% | ~74% | ~74% |

**Best Model: Random Forest** with the highest accuracy, best F1-Score, and most consistent cross-validation performance across all 5 folds.

### Top Predictive Features (Random Forest)

1. Risk Score — custom engineered composite feature
2. Sugar Level — strongest indicator for Diabetes
3. Cholesterol — key cardiovascular marker
4. BP Systolic — flags hypertension and heart conditions
5. Symptom Count — general disease severity measure

---

## Disease Prediction

The system accepts patient data as input and returns a full prediction report:

```
Input  : Patient symptoms + clinical parameters
Output : Predicted disease, risk level, top-3 disease probabilities
```

### Risk Level Classification

| Disease | Risk Level |
|---------|-----------|
| Healthy | LOW |
| Flu | MODERATE |
| Migraine | MODERATE |
| Asthma | HIGH |
| Pneumonia | HIGH |
| Diabetes | HIGH |
| Hypertension | HIGH |
| Heart Disease | CRITICAL |

### Sample Predictions

**Patient 1 — 55-year-old Male**
Symptoms: Fatigue, Dizziness, Frequent Urination, Increased Thirst, Blurred Vision
Sugar Level: 220 | BP: 135/85 | Cholesterol: 195
→ Predicted: **Diabetes** | Risk: HIGH | Confidence: ~78%

**Patient 2 — 28-year-old Female**
Symptoms: Fever, Cough, Fatigue, Body Ache, Headache, Nausea
Sugar Level: 90 | BP: 112/74 | Cholesterol: 165
→ Predicted: **Flu** | Risk: MODERATE | Confidence: ~85%

**Patient 3 — 67-year-old Male**
Symptoms: Chest Pain, Shortness of Breath, Fatigue, Nausea, Dizziness
Sugar Level: 130 | BP: 170/110 | Cholesterol: 280
→ Predicted: **Heart Disease** | Risk: CRITICAL | Confidence: ~82%

---

## Key Insights

| Disease | Key Predictive Indicators |
|---------|--------------------------|
| Diabetes | Frequent Urination + Increased Thirst + Blurred Vision + Sugar > 140 |
| Heart Disease | Chest Pain + Shortness of Breath + Cholesterol > 200 + BP > 140 |
| Hypertension | Systolic BP > 140 + Headache + Dizziness |
| Flu | Fever + Body Ache + Cough + Fatigue |
| Pneumonia | Cough + Fever + Shortness of Breath + Fatigue |
| Migraine | Headache + Nausea + Dizziness without Fever |
| Asthma | Shortness of Breath + Cough without Fever |

---

## Project Structure

```
disease-prediction/
│
├── Disease_Prediction_ML.ipynb    Main Jupyter Notebook
├── dataset.csv                    Patient dataset (2000 records)
├── README.md                      Project documentation
└── requirements.txt               Python dependencies
```

# 📊 Project Outputs


## 🔍 Visualizations

- 📈 [Age Analysis](outputs/age_analysis.png)
- 🏥 [Clinical Metrics](outputs/clinical_metrics.png)
- 🔢 [Confusion Matrices](outputs/confusion_matrices.png)
- 🔗 [Correlation Heatmap](outputs/correlation.png)
- 🔄 [Cross Validation](outputs/cross_validation.png)
- 🦠 [Disease Distribution](outputs/disease_distribution.png)
- ⭐ [Feature Importance](outputs/feature_importance.png)
- 📊 [Final Dashboard](outputs/final_dashboard.png)
- 🚻 [Gender Risk Analysis](outputs/gender_risk.png)

## ⚙️ Model Evaluation

- ❌ [Missing Values](outputs/missing_values.png)
- 📊 [Model Comparison](outputs/model_comparison.png)
- 🔮 [Prediction Probabilities](outputs/prediction_probabilities.png)
- 🌡️ [Symptom Heatmap](outputs/symptom_heatmap.png)

This Disease Prediction System demonstrates how machine learning can meaningfully support healthcare diagnostics. By combining patient symptoms with clinical parameters and engineered risk features, the Random Forest model achieves strong multi-class prediction accuracy across 8 distinct diseases.

The system is designed to be a practical decision-support tool — not a replacement for doctors, but a fast, data-driven assistant that helps prioritize cases and flag high-risk patients early.

---

*Made by Samay Shetty | Glowlogics*
