import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("Day02/data/diabetes.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nOutcome counts:")
print(df['Outcome'].value_counts())

zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print("Missing values after replacing zeros:")
print(df.isnull().sum())

df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

print("\nMissing values after filling:")
print(df.isnull().sum())
print("\nData is clean!")

df['AgeGroup'] = pd.cut(df['Age'], bins=[0, 30, 45, 60, 100], labels=['Under 30', '30-45', '45-60', 'Over 60'])
df['BMICategory'] = pd.cut(df['BMI'], bins=[0, 18.5, 24.9, 29.9, 100], labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

print("New columns added:")
print(df[['Age', 'AgeGroup', 'BMI', 'BMICategory']].head(8))

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7, 5))
df['Outcome'].value_counts().plot(kind='bar',
    color=['#2ECC71', '#E74C3C'], edgecolor='white')
plt.title('Diabetes Outcome Distribution')
plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
plt.ylabel('Number of Patients')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart1_outcome_distribution.png', dpi=150)
plt.show()
print("Chart 1 saved!")

plt.figure(figsize=(7, 5))
for outcome, color, label in [(0,'#2ECC71','No Diabetes'),
                               (1,'#E74C3C','Diabetes')]:
    plt.hist(df[df['Outcome']==outcome]['Glucose'],
             bins=20, alpha=0.6, color=color, label=label, edgecolor='white')
plt.title('Glucose Level Distribution by Outcome')
plt.xlabel('Glucose Level')
plt.ylabel('Number of Patients')
plt.legend()
plt.tight_layout()
plt.savefig('chart2_glucose_distribution.png', dpi=150)
plt.show()
print("Chart 2 saved!")

plt.figure(figsize=(7, 5))
sns.boxplot(data=df, x='Outcome', y='BMI',
            hue='Outcome',
            palette={0: '#2ECC71', 1: '#E74C3C'},
            legend=False)
plt.title('BMI Distribution by Outcome')
plt.xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
plt.ylabel('BMI')
plt.tight_layout()
plt.savefig('chart3_bmi_boxplot.png', dpi=150)
plt.show()
print("Chart 3 saved!")

plt.figure(figsize=(7, 5))
age_diabetes = df.groupby('AgeGroup', observed=True)['Outcome'].mean() * 100
age_diabetes.plot(kind='bar', color='steelblue', edgecolor='white')
plt.title('Diabetes Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Diabetes Rate (%)')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart4_diabetes_by_age.png', dpi=150)
plt.show()
print("Chart 4 saved!")

plt.figure(figsize=(8, 6))
num_cols = df[['Pregnancies','Glucose','BloodPressure',
               'BMI','Age','Insulin','Outcome']]
sns.heatmap(num_cols.corr(), annot=True, fmt=".2f",
            cmap='RdYlGn', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart5_correlation_heatmap.png', dpi=150)
plt.show()
print("Chart 5 saved!")

features = ['Pregnancies', 'Glucose', 'BloodPressure', 'BMI', 'Age', 'Insulin']
X = df[features].copy()
y = df['Outcome'].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("\n== Model Performance ==================================")
print("Accuracy :", round(accuracy * 100, 2), "%")
print("\nDetailed Report:")
print(classification_report(y_test, y_pred,
      target_names=['No Diabetes', 'Diabetes']))