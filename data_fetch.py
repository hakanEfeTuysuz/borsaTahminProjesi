import yfinance as yf
import numpy as np
import pandas as pd
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

# ==========================================
# 1. RASTGELELİĞİ SABİTLEME
# ==========================================
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# ==========================================
# 2. VERİ ÇEKME (End tarihi kapsayıcı yapıldı)
# ==========================================
sembol = "XU100.IS"
print(f"--- {sembol} (BIST 100) V3 Verileri Çekiliyor ---")
# 12 Ağustos verisini garantilemek için bitiş tarihi 13 Ağustos yapıldı
df = yf.download(sembol, start="2020-01-01", end="2026-08-13", progress=False)

kapanis = df['Close'].squeeze()
hacim = df['Volume'].squeeze()
yuksek = df['High'].squeeze()
dusuk = df['Low'].squeeze()

# ==========================================
# 3. PROFESYONEL ÖZELLİK MÜHENDİSLİĞİ (10 Güçlü İndikatör)
# ==========================================
# 1. Getiri ve Momentum
df['Return'] = kapanis.pct_change()
df['Momentum_10'] = kapanis.diff(10)
df['ROC_10'] = kapanis.pct_change(10) * 100

# 2. Hareketli Ortalamalar
df['MA5'] = kapanis.rolling(5).mean()
df['MA20'] = kapanis.rolling(20).mean()
df['MA_Fark'] = (df['MA5'] / df['MA20']) - 1

# 3. RSI
delta = kapanis.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
df['RSI14'] = 100 - (100 / (1 + rs))

# 4. MACD
ema12 = kapanis.ewm(span=12, adjust=False).mean()
ema26 = kapanis.ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

# 5. Bollinger Bands Konumu
bb_mid = kapanis.rolling(20).mean()
bb_std = kapanis.rolling(20).std()
df['BB_Upper'] = bb_mid + (2 * bb_std)
df['BB_Lower'] = bb_mid - (2 * bb_std)
df['BB_Position'] = (kapanis - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)

# 6. ATR (Ortalama Gerçek Aralık - Volatilite Ölçümü)
tr1 = yuksek - dusuk
tr2 = (yuksek - kapanis.shift(1)).abs()
tr3 = (dusuk - kapanis.shift(1)).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR14'] = tr.rolling(14).mean()

# 7. Hacim Oranı
df['Volume_MA20'] = hacim.rolling(20).mean()
df['Volume_Ratio'] = hacim / df['Volume_MA20'].replace(0, np.nan)

# Hedef Değişkenin Hesaplanması
future_close = kapanis.shift(-1)
df['Target'] = np.where(future_close.notna(), (future_close > kapanis).astype(int), np.nan)

df = df.dropna()
df['Target'] = df['Target'].astype(int)

features = [
    'Return', 'Momentum_10', 'ROC_10', 'MA_Fark', 'RSI14', 
    'MACD', 'MACD_Signal', 'BB_Position', 'ATR14', 'Volume_Ratio'
]

X_data = df[features].values
y_data = df['Target'].values

# ==========================================
# 4. ZAMAN BAZLI TRAIN / VAL / TEST BÖLÜNME (Data Leakage Önlemi)
# ==========================================
total_len = len(X_data)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

scaler_x = MinMaxScaler()
# Geleceği görmemesi için scaler SADECE eğitim (train) setiyle eğitilir!
scaler_x.fit(X_data[:train_end])
scaled_X_all = scaler_x.transform(X_data)

def veri_penceresi_olustur(X, y, adim=15):
    X_out, y_out = [], []
    for i in range(adim, len(X)):
        X_out.append(X[i-adim:i, :])
        y_out.append(y[i-1])
    return np.array(X_out), np.array(y_out)

zaman_adimi = 15
X_all_win, y_all_win = veri_penceresi_olustur(scaled_X_all, y_data, zaman_adimi)

# Zaman penceresinden kaynaklı indeks kaymalarını düzeltme
train_idx = train_end - zaman_adimi
val_idx = val_end - zaman_adimi

X_train_lstm = X_all_win[:train_idx]
y_train_lstm = y_all_win[:train_idx]

X_val_lstm = X_all_win[train_idx:val_idx]
y_val_lstm = y_all_win[train_idx:val_idx]

X_test_lstm = X_all_win[val_idx:]
y_test_lstm = y_all_win[val_idx:]

# RF İçin Eşit Şartlar: 15 Gün x 10 Özellik = 150 boyutlu girdi
X_train_rf = X_train_lstm.reshape(X_train_lstm.shape[0], -1)
X_test_rf = X_test_lstm.reshape(X_test_lstm.shape[0], -1)

# ==========================================
# 5. MODELLERİN EĞİTİLMESİ
# ==========================================
print("\n[INFO] Random Forest Eğitiliyor...")
rf_model = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1)
rf_model.fit(X_train_rf, y_train_lstm)
rf_prob = rf_model.predict_proba(X_test_rf)[:, 1]
rf_pred = (rf_prob >= 0.50).astype(int)

print("[INFO] LSTM Eğitiliyor (Açık Validation Seti İle)...")
lstm_model = Sequential([
    LSTM(32, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
    Dropout(0.25),
    Dense(16, activation='relu'),
    Dropout(0.10),
    Dense(1, activation='sigmoid')
])
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lstm_model.fit(
    X_train_lstm, y_train_lstm,
    validation_data=(X_val_lstm, y_val_lstm),
    epochs=50, batch_size=32, shuffle=False, callbacks=[es], verbose=0
)
lstm_prob = lstm_model.predict(X_test_lstm, verbose=0).flatten()
lstm_pred = (lstm_prob >= 0.50).astype(int)

# ==========================================
# 6. GELİŞMİŞ DEĞERLENDİRME METRİKLERİ
# ==========================================
def rapor_yazdir(model_adi, y_true, y_pred, y_prob):
    print(f"\n--- {model_adi} Sonuçları ---")
    print(f"Accuracy  : %{accuracy_score(y_true, y_pred)*100:.1f}")
    print(f"ROC-AUC   : {roc_auc_score(y_true, y_prob):.3f}")
    print(f"Precision : {precision_score(y_true, y_pred, zero_division=0):.3f} (Model 'Yükselecek' dediğinde % kaçı gerçekten yükseldi?)")
    print(f"Recall    : {recall_score(y_true, y_pred, zero_division=0):.3f} (Piyasadaki tüm gerçek yükselişlerin % kaçını yakaladı?)")
    print(f"F1 Score  : {f1_score(y_true, y_pred, zero_division=0):.3f}")
    print(f"Confusion Matrix (Karmaşıklık Matrisi):\n{confusion_matrix(y_true, y_pred)}")

rapor_yazdir("Random Forest", y_test_lstm, rf_pred, rf_prob)
rapor_yazdir("LSTM", y_test_lstm, lstm_pred, lstm_prob)

# ==========================================
# 7. GÖRSELLEŞTİRME (SİNYAL DEĞİŞİM GRAFİĞİ)
# ==========================================
print("\n[INFO] Sinyal grafiği optimize ediliyor...")
gosterilecek_gun = 100

# Test verisinin uzunluğunu baz alarak grafikteki verileri hatasız hizalıyoruz
test_uzunluk = len(rf_pred)
gercek_fiyatlar = kapanis.iloc[-test_uzunluk:]
tarihler = df.index[-test_uzunluk:]

son_fiyatlar = gercek_fiyatlar[-gosterilecek_gun:]
son_tarihler = tarihler[-gosterilecek_gun:]

# Görüntülemek istediğin modelin tahminlerini buradan seçebilirsin (rf_pred veya lstm_pred)
son_tahminler = rf_pred[-gosterilecek_gun:] 

plt.figure(figsize=(14, 7))
plt.plot(son_tarihler, son_fiyatlar, color='black', label='BIST 100 Fiyatı', linewidth=2, alpha=0.7)

# GÜRÜLTÜ FİLTRESİ: Her gün ok basmak yerine, sadece SİNYAL DEĞİŞTİĞİNDE ok koyuyoruz!
for i in range(1, len(son_tahminler)):
    if son_tahminler[i] != son_tahminler[i-1]:
        if son_tahminler[i] == 1:
            plt.scatter(son_tarihler[i], son_fiyatlar.iloc[i] - 120, color='green', marker='^', s=130, zorder=5, label='Al Sinyali' if i == 1 else "")
        else:
            plt.scatter(son_tarihler[i], son_fiyatlar.iloc[i] + 120, color='red', marker='v', s=130, zorder=5, label='Sat Sinyali' if i == 1 else "")

plt.title('BIST 100 - Son 100 Gün: Profesyonel Trend Dönüş Sinyalleri')
plt.xlabel('Tarih')
plt.ylabel('Endeks Puanı (TL)')
plt.grid(True, alpha=0.3)

siyah_cizgi = mlines.Line2D([], [], color='black', label='BIST 100 Fiyatı')
yesil_ok = mlines.Line2D([], [], color='white', markerfacecolor='green', marker='^', markersize=10, label='Trend Dönüşü: Yükseliş (Al)')
kirmizi_ok = mlines.Line2D([], [], color='white', markerfacecolor='red', marker='v', markersize=10, label='Trend Dönüşü: Düşüş (Sat)')
plt.legend(handles=[siyah_cizgi, yesil_ok, kirmizi_ok], loc='upper left')

plt.savefig('bist100_final_sinyal.png')
print("✅ Tertemiz sinyal grafiği 'bist100_final_sinyal.png' olarak güncellendi.")
plt.show()