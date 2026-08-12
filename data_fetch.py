import yfinance as yf
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. RASTGELELİĞİ SABİTLEME
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ==========================================
# 2. VERİ ÇEKME VE BASİT ÖZELLİKLER (KISS Prensibi)
# ==========================================
sembol = "XU100.IS"
print(f"--- {sembol} (BIST 100) Yön Tahmin Verileri Çekiliyor ---")
df = yf.download(sembol, start="2020-01-01", end="2026-08-12", progress=False)

# DataFrame sütunlarını tek boyutlu serilere çevirme (Boyut hatalarına karşı önlem)
kapanis = df['Close'].squeeze()
hacim = df['Volume'].squeeze()

df['Return'] = kapanis.pct_change()
df['MA5'] = kapanis.rolling(5).mean()
df['MA20'] = kapanis.rolling(20).mean()
df['MA_Fark'] = (df['MA5'] - df['MA20']) / df['MA20']
df['Volatilite'] = df['Return'].rolling(10).std()
df['Hacim_Degisim'] = hacim.pct_change()

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df = df.dropna()

# Hedef: Yarın Yükselecek mi? (1: Evet, 0: Hayır)
df['Target'] = (kapanis.shift(-1) > kapanis).astype(int)
df = df.dropna()

features = ['Return', 'MA_Fark', 'Volatilite', 'Hacim_Degisim']
X_data = df[features].values
y_data = df['Target'].values

# ==========================================
# 3. EĞİTİM / TEST AYRIMI VE ÖLÇEKLENDİRME
# ==========================================
egitim_boyutu = int(len(X_data) * 0.8)
X_train_raw, X_test_raw = X_data[:egitim_boyutu], X_data[egitim_boyutu:]
y_train_raw, y_test_raw = y_data[:egitim_boyutu], y_data[egitim_boyutu:]

scaler_x = MinMaxScaler(feature_range=(0, 1))
scaled_X_train = scaler_x.fit_transform(X_train_raw)
scaled_X_test = scaler_x.transform(X_test_raw)

def veri_penceresi_olustur(X, y, adim=15):
    X_out, y_out = [], []
    for i in range(adim, len(X)):
        X_out.append(X[i-adim:i, :])
        y_out.append(y[i-1])
    return np.array(X_out), np.array(y_out)

zaman_adimi = 15
X_train_lstm, y_train_lstm = veri_penceresi_olustur(scaled_X_train, y_train_raw, zaman_adimi)
X_test_lstm, y_test_lstm = veri_penceresi_olustur(scaled_X_test, y_test_raw, zaman_adimi)

# ==========================================
# 4. MODEL EĞİTİMLERİ (LSTM vs RANDOM FOREST)
# ==========================================
print("\n[INFO] Modeller Eğitiliyor...")
lstm_model = Sequential([
    LSTM(units=16, return_sequences=False, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.2),
    Dense(units=8, activation='tanh'),
    Dense(units=1, activation='sigmoid')
])

lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=16, validation_split=0.1, callbacks=[es], verbose=0)
loss, lstm_accuracy = lstm_model.evaluate(X_test_lstm, y_test_lstm, verbose=0)
lstm_model.save('bist100_lstm_model.keras')

X_train_rf = scaled_X_train[zaman_adimi:]
y_train_rf = y_train_raw[zaman_adimi:]
X_test_rf = scaled_X_test[zaman_adimi:]
y_test_rf = y_test_raw[zaman_adimi:]

rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_model.fit(X_train_rf, y_train_rf)
rf_tahminler = rf_model.predict(X_test_rf)
rf_accuracy = accuracy_score(y_test_rf, rf_tahminler)

# ==========================================
# 5. SONUÇLAR VE GÖRSELLEŞTİRME
# ==========================================
print("\n" + "="*50)
print(" 🏆 BIST 100 YÖN TAHMİNİ BENCHMARK SONUÇLARI")
print("="*50)
print(f"🧠 LSTM Başarısı:          %{lstm_accuracy*100:.1f}")
print(f"🌲 Random Forest Başarısı: %{rf_accuracy*100:.1f}")
print("="*50)

print("\n[INFO] Sinyal grafiği oluşturuluyor...")
gosterilecek_gun = 100
test_baslangic_indexi = egitim_boyutu + zaman_adimi

gercek_fiyatlar = kapanis.iloc[test_baslangic_indexi:]
tarihler = df.index[test_baslangic_indexi:]

son_fiyatlar = gercek_fiyatlar[-gosterilecek_gun:]
son_tarihler = tarihler[-gosterilecek_gun:]
son_tahminler = rf_tahminler[-gosterilecek_gun:] 

plt.figure(figsize=(14, 7))
plt.plot(son_tarihler, son_fiyatlar, color='black', label='BIST 100 Fiyatı', linewidth=2, alpha=0.7)

for i in range(len(son_tahminler)):
    if son_tahminler[i] == 1:
        plt.scatter(son_tarihler[i], son_fiyatlar.iloc[i] - 100, color='green', marker='^', s=100, zorder=5)
    else:
        plt.scatter(son_tarihler[i], son_fiyatlar.iloc[i] + 100, color='red', marker='v', s=100, zorder=5)

plt.title('BIST 100 - Son 100 Gün: Yapay Zeka Yön Tahminleri (Random Forest)')
plt.xlabel('Tarih')
plt.ylabel('Endeks Puanı (TL)')
plt.grid(True, alpha=0.3)

siyah_cizgi = mlines.Line2D([], [], color='black', label='BIST 100 Fiyatı')
yesil_ok = mlines.Line2D([], [], color='white', markerfacecolor='green', marker='^', markersize=10, label='Yükseliş Beklentisi')
kirmizi_ok = mlines.Line2D([], [], color='white', markerfacecolor='red', marker='v', markersize=10, label='Düşüş Beklentisi')
plt.legend(handles=[siyah_cizgi, yesil_ok, kirmizi_ok], loc='upper left')

plt.savefig('bist100_final_sinyal.png')
print("✅ İşlem Tamamlandı. Grafik 'bist100_final_sinyal.png' olarak kaydedildi.")
plt.show()