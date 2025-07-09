# kalbar_dashboard_clean.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# 1. Load Data
# =========================
df = pd.read_excel("Data_Kalbar.xlsx")

# =========================
# 2. Visualisasi Tren Iklim
# =========================
plt.figure(figsize=(10, 6))
df.set_index("Tahun")[["Tavg", "kelembaban", "curah_hujan"]].plot()
plt.title("Tren Iklim Kalimantan Barat (Tavg, Kelembaban, Curah Hujan)")
plt.ylabel("Nilai")
plt.xlabel("Tahun")
plt.grid(True)
plt.tight_layout()
plt.savefig("tren_iklim_kalbar.png")
plt.close()

# =========================
# 3. Korelasi Antar Variabel
# =========================
# Ambil hanya kolom numerik
numeric_df = df.select_dtypes(include=['number'])

plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Korelasi Antar Variabel Iklim Kalimantan Barat")
plt.tight_layout()
plt.savefig("korelasi_variabel.png")
plt.close()

# =========================
# 4. Prediksi Curah Hujan
# =========================
# Definisikan fitur dan target
features = ['Tn', 'Tx', 'Tavg', 'kelembaban', 'matahari', 'kecepatan_angin']
target = 'curah_hujan'

# Drop baris dengan NaN (jika ada)
df_clean = df.dropna(subset=features + [target])

# Bagi data
X = df_clean[features]
y = df_clean[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Buat model dan latih
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Cetak hasil
print("===== EVALUASI MODEL RANDOM FOREST =====")
print(f"R² Score  : {r2:.4f}")
print(f"RMSE      : {rmse:.2f} mm")

# Visualisasi aktual vs prediksi
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel("Curah Hujan Aktual (mm)")
plt.ylabel("Curah Hujan Prediksi (mm)")
plt.title("Prediksi vs Aktual Curah Hujan (Random Forest)")
plt.grid(True)
plt.tight_layout()
plt.savefig("prediksi_vs_aktual.png")
plt.close()
