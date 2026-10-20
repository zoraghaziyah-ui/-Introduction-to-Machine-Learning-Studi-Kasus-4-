import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== 1. PERSIAPAN DATASET (DATA-DRIVEN THINKING) ===")
np.random.seed(42)
n_samples = 1000

data = {
    'Income_Annual': np.random.randint(30000, 150000, size=n_samples),
    'Credit_Score': np.random.randint(500, 850, size=n_samples),
    'Loan_Amount': np.random.randint(10000, 80000, size=n_samples),
    'Employment_Years': np.random.randint(0, 15, size=n_samples)
}

df = pd.DataFrame(data)

conditions = (df['Credit_Score'] > 650) & (df['Income_Annual'] > 50000) & (df['Loan_Amount'] < df['Income_Annual'] * 1.5)
df['Loan_Approved'] = np.where(conditions & (np.random.rand(n_samples) > 0.15), 1, 0)

print("5 Baris Pertama Dataset:")
print(df.head(), "\n")

print("=== 2. MEMBAGI DATASET (TRAINING & TESTING) ===")
X = df.drop(columns=['Loan_Approved'])
y = df['Loan_Approved']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Jumlah Data Training: {X_train.shape[0]} baris")
print(f"Jumlah Data Testing : {X_test.shape[0]} baris\n")

print("=== 3. PELATIHAN MODEL (RANDOM FOREST) ===")
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

print("=== 4. EVALUASI PERFORMA MODEL ===")
y_pred = model.predict(X_test)

print("Akurasi Model:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

df_result = X_test.copy()
df_result['Actual_Target'] = y_test
df_result['Predicted_Target'] = y_pred
df_result.to_csv("ml_classification_results.csv", index=False)
print("\nHasil evaluasi model berhasil disimpan ke 'ml_classification_results.csv'")