import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Day01/data/titanic.csv")

print("Shape of dataset (rows, columns):")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nBasic statistics:")
print(df.describe())

print("\nMissing values per column:")
print(df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = df.drop(columns=['Cabin'])

df['Sex_num'] = df['Sex'].map({'male': 0, 'female': 1})

print("Missing values after cleaning:")
print(df.isnull().sum())
print("\nDataset is ready!")

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (8, 5)

plt.figure()
sns.countplot(data=df, x='Sex', hue='Survived', palette=['#E74C3C', '#2ECC71'])
plt.title('Survival Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Number of Passengers')
plt.legend(labels=['Did not survive', 'Survived'])
plt.tight_layout()
plt.savefig('chart1_survival_by_gender.png', dpi=150)
plt.show()
print("Chart 1 saved!")

plt.figure()
plt.hist(df['Age'], bins=20, color='steelblue', edgecolor='white')
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.tight_layout()
plt.savefig('chart2_age_distribution.png', dpi=150)
plt.show()
print("Chart 2 saved!")

plt.figure()
numeric_cols = df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_num']]
correlation = numeric_cols.corr()
sns.heatmap(correlation, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart3_correlation_heatmap.png', dpi=150)
plt.show()
print("Chart 3 saved!")