import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Day03/data/mpg.csv")

print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())
print("\nBasic statistics:")
print(df.describe())

df['horsepower'] = df['horsepower'].fillna(df['horsepower'].median())

df = df.drop(columns=['name'])

df['origin_num'] = df['origin'].map({'usa': 0, 'europe': 1, 'japan': 2})

df['power_to_weight'] = df['horsepower'] / df['weight']

print("Missing values after cleaning:")
print(df.isnull().sum())
print("\nNew columns added:")
print(df[['horsepower', 'weight', 'power_to_weight', 'origin', 'origin_num']].head())

sns.set_theme(style="whitegrid")

plt.figure(figsize=(7,5))
plt.hist(df['mpg'], bins=20, color='steelblue', edgecolor='white')
plt.title('Distribution of MPG (Fuel Efficiency)')
plt.xlabel('Miles Per Gallon')
plt.ylabel('Number of Cars')
plt.tight_layout()
plt.savefig('chart1_mpg_distribution.png', dpi=150)
plt.show()
print("Chart 1 saved!")

plt.figure(figsize=(7,5))
plt.scatter(df['weight'], df['mpg'], alpha=0.6, color='coral')
plt.title('Car Weight vs Fuel Efficiency')
plt.xlabel('Weight (lbs)')
plt.ylabel('MPG')
plt.tight_layout()
plt.savefig('chart2_weight_vs_mpg.png', dpi=150)
plt.show()
print("Chart 2 saved!")

plt.figure(figsize=(7,5))
sns.boxplot(data=df, x='origin', y='mpg', hue='origin',
            palette='Set2', legend=False)
plt.title('Fuel Efficiency by Country of Origin')
plt.xlabel('Origin')
plt.ylabel('MPG')
plt.tight_layout()
plt.savefig('chart3_mpg_by_origin.png', dpi=150)
plt.show()
print("Chart 3 saved!")

plt.figure(figsize=(7,5))
avg_mpg = df.groupby('cylinders')['mpg'].mean()
avg_mpg.plot(kind='bar', color='seagreen', edgecolor='white')
plt.title('Average MPG by Number of Cylinders')
plt.xlabel('Cylinders')
plt.ylabel('Average MPG')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('chart4_mpg_by_cylinders.png', dpi=150)
plt.show()
print("Chart 4 saved!")

plt.figure(figsize=(8,6))
num_cols = df[['mpg','cylinders','displacement','horsepower',
               'weight','acceleration','model_year']]
sns.heatmap(num_cols.corr(), annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('chart5_correlation_heatmap.png', dpi=150)
plt.show()
print("Chart 5 saved!")

features = ['cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model_year', 'origin_num']
X = df[features]
y = df['mpg']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

model1 = LinearRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

model2 = RandomForestRegressor(n_estimators=100, random_state=42)
model2.fit(X_train, y_train)
pred2 = model2.predict(X_test)

def evaluate(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name}")
    print(f"  MAE  : {mae:.2f}")
    print(f"  RMSE : {rmse:.2f}")
    print(f"  R2   : {r2:.2f}")
    return mae, rmse, r2

print("== Model Comparison ==================================")
evaluate("Linear Regression", y_test, pred1)
evaluate("Random Forest Regressor", y_test, pred2)