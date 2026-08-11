import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore') # Kozmetik uyarıları gizle

sembol = "THYAO.IS"
print(f"\n--- {sembol} İçin Gelecek Tahmini Başlatılıyor ---")

# 1. Eğittiğimiz Modeli Yüklüyoruz
model = load_model('borsa_lstm_model.keras')
print("✅ LSTM Modeli başarıyla yüklendi.")

# 2. Ölçeklendiriciyi (Scaler) Yeniden Kuruyoruz
# Gerçek değerlere dönebilmek için eğitimdeki aynı referans aralığına ihtiyacımız var
df_tam = yf.download(sembol, start="2020-01-01", end="2026-08-01", progress=False)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_tam['Close'].values.reshape(-1, 1))

# 3. Modele Soracağımız "Son 60 Günlük" Veriyi Çekiyoruz
# Hisse senedinin bugünkü durumunu bilmesi lazım ki yarını tahmin etsin
df_son = yf.download(sembol, period="3mo", progress=False) 
son_60_gun = df_son['Close'].values[-60:].reshape(-1, 1)

# 4. Veriyi Modelin Anlayacağı Dile Çeviriyoruz
son_60_gun_scaled = scaler.transform(son_60_gun)
X_test = np.array([son_60_gun_scaled])
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 5. BÜYÜK AN: Yapay Zeka Tahmin Yapıyor
tahmin_scaled = model.predict(X_test, verbose=0)
tahmin_gercek_fiyat = scaler.inverse_transform(tahmin_scaled)

print("="*50)
print(f"Yapay Zekanın THYAO.IS İçin Sonraki Gün Kapanış Tahmini: {tahmin_gercek_fiyat[0][0]:.2f} TL")
print("="*50)