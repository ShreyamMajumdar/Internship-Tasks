# 🧠 Day 02 — Data Cleaning, EDA & Machine Learning

![Day](https://img.shields.io/badge/Day-02-purple?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-AI%20%2F%20Data%20Science-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Diabetes-red?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-Logistic%20Regression-orange?style=for-the-badge)

---

## 🎯 Task Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Clean and preprocess a real-world dataset | ✅ Done |
| 2 | Perform detailed Exploratory Data Analysis (EDA) | ✅ Done |
| 3 | Create 5 meaningful visualizations with insights | ✅ Done |
| 4 | Build a simple Machine Learning model | ✅ Done |
| 5 | Document findings and model performance | ✅ Done |

---

## 📂 Dataset Details

| Field | Info |
|-------|------|
| **Name** | Pima Indians Diabetes Dataset |
| **Source** | plotly/datasets (GitHub) |
| **Rows** | 768 |
| **Columns** | 9 |
| **Task Type** | Binary Classification |
| **Target** | Outcome (0 = No Diabetes, 1 = Diabetes) |

### Column Reference

| Column | Type | Description | Issue Found |
|--------|------|-------------|-------------|
| `Pregnancies` | Integer | Number of pregnancies | None |
| `Glucose` | Float | Blood sugar level | ⚠️ Had 0 values |
| `BloodPressure` | Float | Blood pressure | ⚠️ Had 0 values |
| `SkinThickness` | Float | Skin fold thickness | ⚠️ Had 0 values |
| `Insulin` | Float | Insulin level | ⚠️ Had 0 values |
| `BMI` | Float | Body mass index | ⚠️ Had 0 values |
| `DiabetesPedigreeFunction` | Float | Genetic risk score | None |
| `Age` | Integer | Patient age | None |
| `Outcome` | Integer | 0 = No Diabetes, 1 = Diabetes | Target column |

---

## 🛠️ Libraries Used

```python
import pandas as pd                            # Data manipulation
import numpy as np                             # Numerical operations
import matplotlib.pyplot as plt               # Chart drawing
import seaborn as sns                          # Chart styling
from sklearn.model_selection import train_test_split   # Data splitting
from sklearn.linear_model import LogisticRegression   # ML model
from sklearn.metrics import accuracy_score, classification_report  # Evaluation
```

---

## 🔄 Steps Performed

### Step 1 — Data Loading & Exploration

```python
df = pd.read_csv("data/diabetes.csv")
print(df.shape)        # (768, 9)
print(df.describe())   # Statistics
print(df.isnull().sum()) # Checked for missing values
```

**Key findings from exploration:**
- 768 patients, 9 columns
- 268 diabetic (34.9%) vs 500 non-diabetic (65.1%)
- No NaN values shown — but hidden zeros discovered in 5 columns

---

### Step 2 — Data Cleaning

> ⚠️ The dataset showed 0 missing values initially — but columns like Glucose and BMI had **biologically impossible 0 values**, which were actually hidden missing data.

| Column | 0 Values Found | Action Taken |
|--------|----------------|--------------|
| `Glucose` | 5 | Replaced with median |
| `BloodPressure` | 35 | Replaced with median |
| `SkinThickness` | 227 | Replaced with median |
| `Insulin` | 374 | Replaced with median |
| `BMI` | 11 | Replaced with median |

```python
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())
```

---

### Step 3 — Feature Engineering

Two new columns were created to make the data more meaningful for analysis:

```python
# Age Group — bins patients into life stage categories
df['AgeGroup'] = pd.cut(df['Age'],
                         bins=[0, 30, 45, 60, 100],
                         labels=['Under 30', '30-45', '45-60', 'Over 60'])

# BMI Category — standard medical BMI classification
df['BMICategory'] = pd.cut(df['BMI'],
                            bins=[0, 18.5, 24.9, 29.9, 100],
                            labels=['Underweight', 'Normal', 'Overweight', 'Obese'])
```

| New Column | Values | Purpose |
|------------|--------|---------|
| `AgeGroup` | Under 30 / 30–45 / 45–60 / Over 60 | Group patients by life stage |
| `BMICategory` | Underweight / Normal / Overweight / Obese | Standard medical BMI bands |

---

## 📈 Visualizations

### Chart 1 — Diabetes Outcome Distribution
> **File:** `chart1_outcome_distribution.png`

```python
df['Outcome'].value_counts().plot(kind='bar',
    color=['#2ECC71', '#E74C3C'], edgecolor='white')
```

💡 **Insight:** 500 patients (65.1%) have no diabetes vs 268 (34.9%) who do — the dataset is moderately imbalanced.

---

### Chart 2 — Glucose Distribution by Outcome
> **File:** `chart2_glucose_distribution.png`

```python
for outcome, color, label in [(0,'#2ECC71','No Diabetes'),(1,'#E74C3C','Diabetes')]:
    plt.hist(df[df['Outcome']==outcome]['Glucose'],
             bins=20, alpha=0.6, color=color, label=label)
```

💡 **Insight:** Diabetic patients have significantly higher glucose levels, mostly above 120. Non-diabetic patients cluster around 90–110.

---

### Chart 3 — BMI Distribution by Outcome
> **File:** `chart3_bmi_boxplot.png`

```python
sns.boxplot(data=df, x='Outcome', y='BMI',
            palette={0: '#2ECC71', 1: '#E74C3C'})
```

💡 **Insight:** Diabetic patients have a higher median BMI (~35) compared to non-diabetic patients (~30), suggesting obesity as a risk factor.

---

### Chart 4 — Diabetes Rate by Age Group
> **File:** `chart4_diabetes_by_age.png`

```python
age_diabetes = df.groupby('AgeGroup', observed=True)['Outcome'].mean() * 100
age_diabetes.plot(kind='bar', color='steelblue', edgecolor='white')
```

💡 **Insight:** Diabetes rate rises significantly with age — from ~28% in the under-30 group to over ~55% in the over-60 group.

---

### Chart 5 — Correlation Heatmap
> **File:** `chart5_correlation_heatmap.png`

```python
sns.heatmap(num_cols.corr(), annot=True, fmt=".2f", cmap='RdYlGn')
```

💡 **Insight:** `Glucose` has the highest correlation with `Outcome` (0.49), confirming it as the most important predictor. `BMI` and `Age` also show meaningful correlations.

---

## 🤖 Machine Learning Model

### Model Configuration

| Setting | Value |
|---------|-------|
| **Algorithm** | Logistic Regression |
| **Features Used** | Pregnancies, Glucose, BloodPressure, BMI, Age, Insulin |
| **Target** | Outcome |
| **Train Split** | 80% (614 samples) |
| **Test Split** | 20% (154 samples) |
| **Random State** | 42 |

### Model Code

```python
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']
X = df[features]
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Model Performance

| Metric | No Diabetes | Diabetes | Overall |
|--------|-------------|----------|---------|
| **Precision** | 0.81 | 0.69 | — |
| **Recall** | 0.86 | 0.62 | — |
| **F1-Score** | 0.83 | 0.65 | — |
| **Accuracy** | — | — | **~77%** |

---

## 🔍 Key Findings

```
✦ Glucose level is the strongest predictor of diabetes (correlation: 0.49)
✦ Patients with diabetes have higher average BMI (~35 vs ~30)
✦ Diabetes risk increases significantly with age
✦ 374 out of 768 insulin values were hidden zeros — required careful cleaning
✦ Logistic Regression achieved ~77% accuracy on unseen test data
✦ The model predicts non-diabetic patients more reliably than diabetic ones
```

---

## 📚 Learning Outcomes

- Learned to detect and handle **hidden missing values** disguised as zeros
- Applied **feature engineering** to create meaningful new columns
- Created **5 different chart types** covering distribution, comparison, and correlation
- Built and evaluated a **Logistic Regression classifier** using scikit-learn
- Understood **train/test splitting** and basic model evaluation metrics

---

## ▶️ How to Run

```bash
# Make sure you're in the Day02 folder
cd Day02

# Run the analysis
python analysis.py

# Expected output:
# Data is clean!
# Chart 1 saved!
# Chart 2 saved!
# Chart 3 saved!
# Chart 4 saved!
# Chart 5 saved!
# Accuracy: ~77%
```

---

## 📁 Folder Structure

```
Day02/
├── data/
│   └── diabetes.csv
├── chart1_outcome_distribution.png
├── chart2_glucose_distribution.png
├── chart3_bmi_boxplot.png
├── chart4_diabetes_by_age.png
├── chart5_correlation_heatmap.png
├── analysis.py
└── README.md
```

---

> ⬅️ [Back to Day 01](../Day01/README.md) &nbsp;&nbsp;|&nbsp;&nbsp; 🏠 [Main README](../README.md)