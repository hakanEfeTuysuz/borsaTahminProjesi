import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

sembol = "THYAO.IS"
# KULLANICI ARAYÜZÜNDEN (UI) GELECEK OLAN DEĞER BURASI:
istenen_gun_sayisi = 3  # Burayı 3, 10, veya 34 yaparak test edebilirsin

print(f"\n--- {sembol} İçin {istenen_gun_sayisi} Günlük Gelecek Simülasyonu Başlatılıyor ---")

# 1. Modeli ve Scaler'ı Hazırlama
model = load_model('borsa_lstm_model.keras')
df_tam = yf.download(sembol, start="2020-01-01", end="2026-08-01", progress=False)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(df_tam['Close'].values.reshape(-1, 1))

# 2. Son 60 Günün Gerçek Verisini Çekme (Grafikte göstermek için biraz daha fazla çekiyoruz)
df_son = yf.download(sembol, period="6mo", progress=False) 
gercek_fiyatlar = df_son['Close'].values.flatten()
gercek_tarihler = df_son.index

# Modele beslenecek son 60 günlük pencere
son_60_gun = gercek_fiyatlar[-60:].reshape(-1, 1)
mevcut_pencere_scaled = scaler.transform(son_60_gun)

# 3. ZİNCİRLEME (RECURSIVE) TAHMİN DÖNGÜSÜ
gelecek_tahminler_scaled = []

for _ in range(istenen_gun_sayisi):
    # Mevcut 60 günü modele ver
    X_test = np.reshape(mevcut_pencere_scaled, (1, 60, 1))
    
    # Yeni günü tahmin et
    yeni_tahmin = model.predict(X_test, verbose=0)
    gelecek_tahminler_scaled.append(yeni_tahmin[0, 0])
    
    # Diziyi Kaydır: En eski günü at, yeni tahmini sıranın sonuna ekle!
    mevcut_pencere_scaled = np.append(mevcut_pencere_scaled[1:], yeni_tahmin)
    mevcut_pencere_scaled = mevcut_pencere_scaled.reshape(-1, 1)

# Tahminleri gerçek TL fiyatlarına geri çevir
gelecek_tahminler_gercek = scaler.inverse_transform(np.array(gelecek_tahminler_scaled).reshape(-1, 1)).flatten()

print("="*50)
print(f"🔮 Gelecek {istenen_gun_sayisi} Günlük Kapanış Beklentileri:")
for i, fiyat in enumerate(gelecek_tahminler_gercek, 1):
    print(f"{i}. Gün Tahmini: {fiyat:.2f} TL")
print("="*50)

# 4. VERİLERİ GÖRSELLEŞTİRME (MATPLOTLIB SİHRİ)
plt.figure(figsize=(14, 6))

# A. Gerçek verilerin son 100 gününü çiz
gosterilecek_gecmis = 100
plt.plot(range(gosterilecek_gecmis), gercek_fiyatlar[-gosterilecek_gecmis:], color='blue', label='Gerçek Kapanış Fiyatları')

# B. Tahmin edilen verileri çiz (Gerçek verinin bittiği yerden başlayacak şekilde)
tahmin_x_ekseni = range(gosterilecek_gecmis - 1, gosterilecek_gecmis + istenen_gun_sayisi)
# Kırılma olmasın diye son gerçek fiyatı tahmin dizisinin başına ekliyoruz
birlestirilmis_tahmin = np.insert(gelecek_tahminler_gercek, 0, gercek_fiyatlar[-1])

plt.plot(tahmin_x_ekseni, birlestirilmis_tahmin, color='red', linestyle='dashed', marker='o', label=f'Yapay Zeka Tahmini ({istenen_gun_sayisi} Gün)')

plt.title(f'{sembol} - LSTM Gelecek Projeksiyonu')
plt.xlabel('Zaman Çizelgesi (Gün)')
plt.ylabel('Fiyat (TL)')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafiği Ekrana Bas
print("📊 Grafik oluşturuldu! Lütfen açılan pencereyi kontrol edin.")
plt.show()