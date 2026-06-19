import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(os.path.join(BASE_DIR, "data", "churn.csv"))

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nChurn distribution:")
print(df['Churn'].value_counts())

df = df.drop(columns=['customerID'])

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum(), "missing values remaining")

df['avg_monthly_spend'] = df['TotalCharges'] / (df['tenure'] + 1)

service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
df['num_services'] = df[service_cols].apply( lambda row: sum(1 for v in row if v == 'Yes'), axis=1)

print("\nFeature Engineering done:")
print(df[['tenure', 'TotalCharges', 'avg_monthly_spend', 'num_services']].head())

le = LabelEncoder()

cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols: df[col] = le.fit_transform(df[col])

print("\nAll categorical columns encoded.")
print("Churn value counts (0=No, 1=Yes):")
print(df['Churn'].value_counts())

X = df.drop(columns=['Churn'])
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

def make_pipeline(model):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model',  model)
    ])

models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
    "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=100, random_state=42)),
    "Gradient Boosting": make_pipeline(GradientBoostingClassifier(random_state=42)),
    "KNN": make_pipeline(KNeighborsClassifier(n_neighbors=5)),
}

results = {}
print("\n== Final Model Comparison ====================================")
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    test_acc = accuracy_score(y_test, pred)
    cv_scores = cross_val_score(pipe, X, y, cv=5)
    results[name] = {
        "test_acc": round(test_acc, 2),
        "cv_mean": round(cv_scores.mean(), 2),
        "pred": pred
    }
    print(f"\n{name}")
    print(f"Test Accuracy : {round(test_acc, 2)}")
    print(f"CV Mean Acc : {round(cv_scores.mean(), 2)}")

best_model = max(results, key=lambda n: results[n]['cv_mean'])
print(f"\nBest model: {best_model} ({results[best_model]['cv_mean']} CV accuracy)")

print(f"\nDetailed Report — {best_model}:")
print(classification_report(y_test, results[best_model]['pred'],
      target_names=['No Churn', 'Churn']))

df['Churn_label'] = df['Churn'].map({0: 'No', 1: 'Yes'})

print("\n== Automated Business Insights ==============================")
print(f"Total customers : {len(df)}")
print(f"Overall churn rate : {round(df['Churn'].mean()*100, 1)}%")
print(f"Avg tenure (stay) : {round(df[df['Churn']==0]['tenure'].mean(), 1)} months")
print(f"Avg tenure (churn) : {round(df[df['Churn']==1]['tenure'].mean(), 1)} months")
print(f"Avg monthly (stay) : ${round(df[df['Churn']==0]['MonthlyCharges'].mean(), 2)}")
print(f"Avg monthly (churn) : ${round(df[df['Churn']==1]['MonthlyCharges'].mean(), 2)}")
print(f"Avg services (stay) : {round(df[df['Churn']==0]['num_services'].mean(), 1)}")
print(f"Avg services (churn) : {round(df[df['Churn']==1]['num_services'].mean(), 1)}")
print(f"Best CV model : {best_model} ({results[best_model]['cv_mean']})")

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
churn_counts = df['Churn'].value_counts()
plt.bar(['No Churn', 'Churn'], churn_counts.values,
        color=['#2ECC71', '#E74C3C'], edgecolor='white')
for i, v in enumerate(churn_counts.values):
    plt.text(i, v + 50, str(v), ha='center', fontweight='bold')
plt.title('Customer Churn Distribution')
plt.xlabel('Churn Status')
plt.ylabel('Number of Customers')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart1_churn_distribution.png'), dpi=150)
plt.show()
print("Chart 1 saved!")

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges',
            hue='Churn', palette={0: '#2ECC71', 1: '#E74C3C'}, legend=False)
plt.title('Monthly Charges by Churn Status')
plt.xlabel('Churn (0 = No, 1 = Yes)')
plt.ylabel('Monthly Charges ($)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart2_monthly_charges_churn.png'), dpi=150)
plt.show()
print("Chart 2 saved!")

plt.figure(figsize=(7, 5))
for churn_val, color, label in [(0, '#2ECC71', 'No Churn'), (1, '#E74C3C', 'Churn')]:
    plt.hist(df[df['Churn'] == churn_val]['tenure'],
             bins=25, alpha=0.6, color=color, label=label, edgecolor='white')
plt.title('Tenure Distribution by Churn Status')
plt.xlabel('Tenure (Months)')
plt.ylabel('Number of Customers')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart3_tenure_distribution.png'), dpi=150)
plt.show()
print("Chart 3 saved!")

plt.figure(figsize=(7, 5))
contract_churn = df.groupby('Contract')['Churn'].mean() * 100
contract_labels = ['Month-to-Month', 'One Year', 'Two Year']
plt.bar(contract_labels, contract_churn.values,
        color=['#E74C3C', '#F39C12', '#2ECC71'], edgecolor='white')
for i, v in enumerate(contract_churn.values):
    plt.text(i, v + 0.5, f"{round(v, 1)}%", ha='center', fontweight='bold')
plt.title('Churn Rate by Contract Type')
plt.xlabel('Contract Type')
plt.ylabel('Churn Rate (%)')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart4_churn_by_contract.png'), dpi=150)
plt.show()
print("Chart 4 saved!")

plt.figure(figsize=(7, 5))
service_churn = df.groupby('num_services')['Churn'].mean() * 100
service_churn.plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Churn Rate by Number of Services Subscribed')
plt.xlabel('Number of Add-On Services')
plt.ylabel('Churn Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart5_churn_by_services.png'), dpi=150)
plt.show()
print("Chart 5 saved!")

plt.figure(figsize=(12, 9))
numeric_df = df[['tenure', 'MonthlyCharges', 'TotalCharges',
                 'avg_monthly_spend', 'num_services',
                 'SeniorCitizen', 'Churn']].copy()
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f",
            cmap='coolwarm', linewidths=0.5)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart6_correlation_heatmap.png'), dpi=150)
plt.show()
print("Chart 6 saved!")

plt.figure(figsize=(9, 5))
names    = list(results.keys())
cv_means = [results[n]['cv_mean'] for n in names]
colors   = ['#3498DB', '#2ECC71', '#E67E22', '#9B59B6']
bars = plt.bar(names, cv_means, color=colors, edgecolor='white')
for bar, val in zip(bars, cv_means):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 0.003, str(val),
             ha='center', fontweight='bold')
plt.title('Final Model Comparison — 5-Fold CV Accuracy')
plt.ylabel('CV Mean Accuracy')
plt.ylim(0, 1)
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'chart7_model_comparison.png'), dpi=150)
plt.show()
print("Chart 7 saved!")