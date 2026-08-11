import yfinance as yf
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

# 1. RASTGELELİĞİ SABİTLEME (Tekrarlanabilirlik İçin)
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

sembol = "THYAO.IS"
print(f"--- {sembol} Verileri Çekiliyor ---")
df = yf.download(sembol, start="2020-01-01", end="2026-08-01", progress=False)

# Hata Kontrolü
if df.empty:
    raise ValueError(f"HATA: {sembol} verisi çekilemedi. İnternet bağlantınızı veya sembolü kontrol edin.")

dataset = df['Close'].values.reshape(-1, 1)

# 2. TRAIN/TEST AYRIMI (%80 Eğitim, %20 Test)
# Modeli daha önce hiç görmediği veriyle test etmek zorundayız!
egitim_boyutu = int(len(dataset) * 0.8)
egitim_verisi = dataset[:egitim_boyutu]
test_verisi = dataset[egitim_boyutu:]

# 3. VERİ SIZINTISINI (DATA LEAKAGE) ÖNLEME
# Scaler SADECE eğitim verisiyle eğitilir (fit). Test verisine sadece uygulanır (transform).
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_egitim = scaler.fit_transform(egitim_verisi)
scaled_test = scaler.transform(test_verisi)

# Zaman Serisi Penceresi Oluşturma Fonksiyonu
def veri_penceresi_olustur(veri, zaman_adimi=60):
    X, y = [], []
    for i in range(zaman_adimi, len(veri)):
        X.append(veri[i-zaman_adimi:i, 0])
        y.append(veri[i, 0])
    return np.array(X), np.array(y)

zaman_adimi = 60
X_train, y_train = veri_penceresi_olustur(scaled_egitim, zaman_adimi)
X_test, y_test = veri_penceresi_olustur(scaled_test, zaman_adimi)

# 3D LSTM Formatına Çevirme
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print("\n--- LSTM Modeli Kuruluyor ---")
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. ERKEN DURDURMA (EARLY STOPPING)
# Model ezberlemeye (overfitting) başlarsa 50 tur beklemez, eğitimi en iyi yerde durdurur.
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Model başarıyla derlendi. Eğitim başlıyor...\n")
history = model.fit(X_train, y_train, epochs=50, batch_size=32, 
                    validation_split=0.1, callbacks=[es])
print("\nModel eğitimi tamamlandı!")

# 5. GERÇEK SINAV: METRİKLERLE DEĞERLENDİRME
print("\n--- Test Seti Değerlendirmesi ---")
tahminler_scaled = model.predict(X_test, verbose=0)
tahminler_gercek = scaler.inverse_transform(tahminler_scaled)
y_test_gercek = scaler.inverse_transform(y_test.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_gercek, tahminler_gercek))
mae = mean_absolute_error(y_test_gercek, tahminler_gercek)

print(f"RMSE (Kök Ortalama Kare Hata): {rmse:.2f} TL")
print(f"MAE (Ortalama Mutlak Hata):    {mae:.2f} TL")

# Fiyat Aralığı ve Yüzdesel Hata (Claude'un Önerisi)
ortalama_fiyat = y_test_gercek.mean()
print(f"Test verisi fiyat aralığı:     {y_test_gercek.min():.2f} - {y_test_gercek.max():.2f} TL")
print(f"MAE'nin ortalama fiyata oranı: %{(mae/ortalama_fiyat)*100:.2f}")

# 6. NAIVE BASELINE KARŞILAŞTIRMASI (Gerçeklik Testi)
# "Yarın, bugünle aynı olacak" diyen basit bir modelin hatası
naive_tahmin = y_test_gercek[:-1]  
naive_gercek = y_test_gercek[1:]

naive_rmse = np.sqrt(mean_squared_error(naive_gercek, naive_tahmin))
naive_mae = mean_absolute_error(naive_gercek, naive_tahmin)

print(f"\n--- Naive Baseline (Referans Çizgisi) ---")
print(f"Naive RMSE: {naive_rmse:.2f} TL")
print(f"Naive MAE:  {naive_mae:.2f} TL")

if mae < naive_mae:
    print("🏆 Başarılı: LSTM modelimiz basit referans çizgisini yendi ve gerçekten bir şeyler öğrendi!")
else:
    print("⚠️ Uyarı: LSTM modelimiz 'dünü kopyala' stratejisinden daha iyi bir sonuç üretemedi. Model mimarisini veya özelliklerini (features) geliştirmek gerekebilir.")

# 7. YÖN DOĞRULUĞU (Directional Accuracy)
gercek_yon = np.diff(y_test_gercek.flatten()) > 0
tahmin_yon = np.diff(tahminler_gercek.flatten()) > 0
yon_dogrulugu = np.mean(gercek_yon == tahmin_yon) * 100
print(f"\n📈 Yön Tahmin Doğruluğu: %{yon_dogrulugu:.1f}")

# Modeli Kaydet
model.save('borsa_lstm_model.keras')
print("\n✅ Model ağırlıkları 'borsa_lstm_model.keras' olarak kaydedildi.")


# 8. VERİ GÖRSELLEŞTİRME VE KAYDETME
import matplotlib.pyplot as plt

plt.figure(figsize=(14,6))
plt.plot(y_test_gercek, label='Gerçek Fiyat', color='blue')
plt.plot(tahminler_gercek, label='Tahmin Edilen Fiyat', color='red', alpha=0.7)
plt.title(f'{sembol} - Gerçek vs Tahmin (Test Seti)')
plt.xlabel('Gün')
plt.ylabel('Fiyat (TL)')
plt.legend()
plt.grid(True, alpha=0.3)

# Grafiği kaydet (README için kullanılacak)
plt.savefig('tahmin_grafigi.png')
print("\n📊 Grafik 'tahmin_grafigi.png' olarak kaydedildi. Ekranda açılıyor...")
plt.show()