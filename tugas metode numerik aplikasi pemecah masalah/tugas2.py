import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Membaca data
data = pd.read_csv('Student_Performance.csv')

# Menyiapkan data untuk analisis
X = data['Hours Studied'].values.reshape(-1, 1)
y = data['Performance Index'].values

# Regresi Linear (Metode 1)
model1 = LinearRegression()
model1.fit(X, y)

a1, b1 = model1.coef_[0], model1.intercept_
print(f"\nModel Linear (Metode 1):")
print(f"Performance Index = {a1:.2f} * Hours Studied + {b1:.2f}")
print(f"Interpretasi: Setiap penambahan 1 jam belajar, Performance Index meningkat sebesar {a1:.2f} poin.")

# Regresi Pangkat (Metode 2)
# Menghindari log(0) dengan menambahkan konstanta kecil
epsilon = 1e-10
X_log = np.log(X.ravel() + epsilon)  # Menggunakan .ravel() untuk 1D array
y_log = np.log(y + epsilon)

model2 = LinearRegression()
model2.fit(X_log.reshape(-1, 1), y_log)

log_a2, b2 = model2.intercept_, model2.coef_[0]
a2 = np.exp(log_a2)
print(f"\nModel Pangkat (Metode 2):")
print(f"Performance Index = {a2:.2f} * (Hours Studied)^{b2:.2f}")
print(f"Interpretasi: Setiap kenaikan 1% jam belajar, Performance Index meningkat sekitar {b2:.2f}%.")

# Visualisasi
plt.figure(figsize=(12, 6))

# Plot data asli
plt.scatter(X, y, label='Data Asli', color='blue', alpha=0.7)

# Plot model linear
X_range = np.linspace(X.min(), X.max(), 100)
y_pred1 = a1 * X_range + b1
plt.plot(X_range, y_pred1, label='Model Linear', color='red')

# Plot model pangkat
X_range_log = np.log(X_range + epsilon)
y_pred2 = a2 * np.exp(b2 * X_range_log)
plt.plot(X_range, y_pred2, label='Model Pangkat', color='green')

plt.xlabel('Durasi Belajar (Jam)')
plt.ylabel('Indeks Performa')
plt.title('Hubungan Durasi Belajar dengan Indeks Performa Siswa')
plt.legend()
plt.grid(True)
plt.show()

# Analisis tambahan: Korelasi
correlation = np.corrcoef(X.flatten(), y)[0, 1]
print(f"\nKorelasi antara Hours Studied dan Performance Index: {correlation:.2f}")
print("Interpretasi Korelasi:")
if correlation > 0.8:
    print("Korelasi sangat kuat positif: Durasi belajar sangat berpengaruh terhadap indeks performa.")
elif correlation > 0.6:
    print("Korelasi kuat positif: Durasi belajar berpengaruh signifikan terhadap indeks performa.")
elif correlation > 0.4:
    print("Korelasi sedang positif: Durasi belajar cukup berpengaruh terhadap indeks performa.")
elif correlation > 0.2:
    print("Korelasi lemah positif: Durasi belajar sedikit berpengaruh terhadap indeks performa.")
else:
    print("Korelasi sangat lemah atau tidak ada: Faktor lain mungkin lebih berpengaruh.")

# Analisis Residual untuk menilai kecocokan model
residuals1 = y - model1.predict(X)

X_reshaped = X.ravel() + epsilon
residuals2 = y - a2 * np.exp(b2 * np.log(X_reshaped))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(model1.predict(X), residuals1, color='red', alpha=0.7)
plt.xlabel('Prediksi Model Linear')
plt.ylabel('Residual')
plt.title('Residual Plot - Model Linear')
plt.grid(True)

plt.subplot(1, 2, 2)
y_pred2 = a2 * np.exp(b2 * np.log(X_reshaped))
plt.scatter(y_pred2, residuals2, color='green', alpha=0.7)
plt.xlabel('Prediksi Model Pangkat')
plt.ylabel('Residual')
plt.title('Residual Plot - Model Pangkat')
plt.grid(True)

plt.tight_layout()
plt.show()

print("\nAnalisis Residual:")
print("- Jika residual tersebar acak sekitar 0, model cocok dengan data.")
print("- Jika ada pola (misalnya, kurva), model mungkin tidak menangkap semua informasi.")