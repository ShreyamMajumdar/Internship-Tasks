# 📊 Day 01 — Exploratory Data Analysis

![Day](https://img.shields.io/badge/Day-01-blue?style=for-the-badge)
![Domain](https://img.shields.io/badge/Domain-AI%20%2F%20Data%20Science-blue?style=for-the-badge&logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-Titanic-orange?style=for-the-badge)

---

## 🎯 Task Objectives

| # | Objective | Status |
|---|-----------|--------|
| 1 | Analyse a given CSV dataset | ✅ Done |
| 2 | Perform basic data exploration and preprocessing | ✅ Done |
| 3 | Create and present 3 meaningful visualizations with insights | ✅ Done |

---

## 📂 Dataset Details

| Field | Info |
|-------|------|
| **Name** | Titanic Passenger Dataset |
| **Source** | datasciencedojo/datasets (GitHub) |
| **Rows** | 891 |
| **Columns** | 12 |
| **Target** | Survived (0 = No, 1 = Yes) |

### Column Reference

| Column | Type | Description | Missing? |
|--------|------|-------------|----------|
| `PassengerId` | Integer | Unique passenger ID | No |
| `Survived` | Integer | Survival: 0 = No, 1 = Yes | No |
| `Pclass` | Integer | Ticket class (1, 2, 3) | No |
| `Name` | String | Full name | No |
| `Sex` | String | Gender (male / female) | No |
| `Age` | Float | Age in years | ⚠️ 177 missing |
| `SibSp` | Integer | Siblings / spouses aboard | No |
| `Parch` | Integer | Parents / children aboard | No |
| `Ticket` | String | Ticket number | No |
| `Fare` | Float | Fare paid (£) | No |
| `Cabin` | String | Cabin number | ⚠️ 687 missing |
| `Embarked` | String | Port: C / Q / S | ⚠️ 2 missing |

---

## 🛠️ Libraries Used

```python
import pandas as pd        # Data loading and manipulation
import numpy as np         # Numerical operations
import matplotlib.pyplot as plt  # Chart drawing
import seaborn as sns      # Chart styling
```

---

## 🔄 Steps Performed

### Step 1 — Data Loading & Exploration

```python
df = pd.read_csv("data/titanic.csv")
print(df.shape)        # (891, 12)
print(df.head())       # First 5 rows
print(df.describe())   # Statistics
print(df.isnull().sum()) # Missing values
```

**Key findings from exploration:**
- 891 passengers total
- Only 38.4% survived (342 out of 891)
- 577 male vs 314 female passengers
- 3 columns had missing values

---

### Step 2 — Data Preprocessing

| Step | Action | Reason |
|------|--------|--------|
| Fill `Age` | Replaced 177 NaN with median (28.0) | Median is less affected by outliers |
| Fill `Embarked` | Replaced 2 NaN with mode ('S') | Only 2 missing — safest fix |
| Drop `Cabin` | Removed entire column | 77% values missing — unusable |
| Encode `Sex` | male → 0, female → 1 | Required for correlation analysis |

```python
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df = df.drop(columns=['Cabin'])
df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1})
```

---

## 📈 Visualizations

### Chart 1 — Survival Count by Gender
> **File:** `chart1_survival_by_gender.png`

```python
sns.countplot(data=df, x='Sex', hue='Survived',
              palette=['#E74C3C', '#2ECC71'])
```

💡 **Insight:** Women survived at ~74% vs only ~18% for men — reflecting the "women and children first" policy.

---

### Chart 2 — Age Distribution of Passengers
> **File:** `chart2_age_distribution.png`

```python
plt.hist(df['Age'], bins=20, color='steelblue', edgecolor='white')
```

💡 **Insight:** Most passengers were between 20–40 years old. A small peak near age 0–5 represents young children.

---

### Chart 3 — Correlation Heatmap
> **File:** `chart3_correlation_heatmap.png`

```python
sns.heatmap(numeric_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm')
```

💡 **Insight:** `Sex_num` (0.54) and `Pclass` (-0.34) are the strongest predictors of survival.

---

## 🔍 Key Findings

```
✦ Only 38.4% of passengers survived the disaster
✦ Women survived at 74% vs men at only 18%
✦ Most passengers were aged between 20–40 years
✦ Cabin data was 77% missing — too unreliable to use
✦ Higher ticket class = better survival chances
✦ Fare and Sex are the strongest survival predictors
```

---

## ▶️ How to Run

```bash
# Make sure you're in the Day01 folder
cd Day01

# Run the analysis
python analysis.py

# Expected output:
# Shape: (891, 12)
# Missing values after cleaning: 0
# Chart 1 saved!
# Chart 2 saved!
# Chart 3 saved!
```

---

## 📁 Folder Structure

```
Day01/
├── data/
│   └── titanic.csv
├── chart1_survival_by_gender.png
├── chart2_age_distribution.png
├── chart3_correlation_heatmap.png
├── analysis.py
└── README.md
```

---

> ⬅️ [Back to Main README](../README.md) &nbsp;&nbsp;|&nbsp;&nbsp; ➡️ [Day 02](../Day02/README.md)