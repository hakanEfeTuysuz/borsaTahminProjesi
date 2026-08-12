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
# 2. VERİ ÇEKME
# ==========================================
sembol = "XU100.IS"
print(f"--- {sembol} (BIST 100) V3 Verileri Çekiliyor ---")
df = yf.download(sembol, start="2020-01-01", end="2026-08-13", progress=False)

kapanis = df['Close'].squeeze()
hacim = df['Volume'].squeeze()
yuksek = df['High'].squeeze()
dusuk = df['Low'].squeeze()

# ==========================================
# 3. PROFESYONEL ÖZELLİK MÜHENDİSLİĞİ (10 Güçlü İndikatör)
# ==========================================
df['Return'] = kapanis.pct_change()
df['Momentum_10'] = kapanis.diff(10)
df['ROC_10'] = kapanis.pct_change(10) * 100

df['MA5'] = kapanis.rolling(5).mean()
df['MA20'] = kapanis.rolling(20).mean()
df['MA_Fark'] = (df['MA5'] / df['MA20']) - 1

delta = kapanis.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
df['RSI14'] = 100 - (100 / (1 + rs))

ema12 = kapanis.ewm(span=12, adjust=False).mean()
ema26 = kapanis.ewm(span=26, adjust=False).mean()
df['MACD'] = ema12 - ema26
df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

bb_mid = kapanis.rolling(20).mean()
bb_std = kapanis.rolling(20).std()
df['BB_Upper'] = bb_mid + (2 * bb_std)
df['BB_Lower'] = bb_mid - (2 * bb_std)
df['BB_Position'] = (kapanis - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower']).replace(0, np.nan)

tr1 = yuksek - dusuk
tr2 = (yuksek - kapanis.shift(1)).abs()
tr3 = (dusuk - kapanis.shift(1)).abs()
tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
df['ATR14'] = tr.rolling(14).mean()

df['Volume_MA20'] = hacim.rolling(20).mean()
df['Volume_Ratio'] = hacim / df['Volume_MA20'].replace(0, np.nan)

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
# 4. ZAMAN BAZLI TRAIN / VAL / TEST BÖLÜNME
# ==========================================
total_len = len(X_data)
train_end = int(total_len * 0.70)
val_end = int(total_len * 0.85)

scaler_x = MinMaxScaler()
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

train_idx = train_end - zaman_adimi
val_idx = val_end - zaman_adimi

X_train_lstm = X_all_win[:train_idx]
y_train_lstm = y_all_win[:train_idx]

X_val_lstm = X_all_win[train_idx:val_idx]
y_val_lstm = y_all_win[train_idx:val_idx]

X_test_lstm = X_all_win[val_idx:]
y_test_lstm = y_all_win[val_idx:]

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
# 7. GÖRSELLEŞTİRME (DÜZELTİLDİ)
# ==========================================
print("\n[INFO] Detaylı tahmin grafiği hazırlanıyor...")
gosterilecek_gun = 100

# Test verisinin gerçek başlangıç noktası val_end'dir
gercek_fiyatlar = kapanis.iloc[val_end:]
tarihler = df.index[val_end:]

son_fiyatlar = gercek_fiyatlar[-gosterilecek_gun:]
son_tarihler = tarihler[-gosterilecek_gun:]

# Zaten yukarıda hesaplanan tahminlerin son kısımlarını çekiyoruz
son_rf_tahminler = rf_pred[-gosterilecek_gun:]
son_lstm_tahminler = lstm_pred[-gosterilecek_gun:]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- 1. ÜST GRAFİK: RANDOM FOREST ---
ax1.plot(son_tarihler, son_fiyatlar, color='black', label='BIST 100 Fiyatı', linewidth=1.5, alpha=0.6)
for i in range(len(son_rf_tahminler)):
    if son_rf_tahminler[i] == 1:
        ax1.scatter(son_tarihler[i], son_fiyatlar.iloc[i] - 80, color='green', marker='^', s=40, alpha=0.8)
    else:
        ax1.scatter(son_tarihler[i], son_fiyatlar.iloc[i] + 80, color='red', marker='v', s=40, alpha=0.8)

ax1.set_title('BIST 100 - Random Forest (Geleneksel ML) Günlük Tahmin Dağılımı')
ax1.set_ylabel('Endeks Puanı (TL)')
ax1.grid(True, alpha=0.3)

# --- 2. ALT GRAFİK: LSTM ---
ax2.plot(son_tarihler, son_fiyatlar, color='black', label='BIST 100 Fiyatı', linewidth=1.5, alpha=0.6)
for i in range(len(son_lstm_tahminler)):
    if son_lstm_tahminler[i] == 1:
        ax2.scatter(son_tarihler[i], son_fiyatlar.iloc[i] - 80, color='blue', marker='^', s=40, alpha=0.8)
    else:
        ax2.scatter(son_tarihler[i], son_fiyatlar.iloc[i] + 80, color='orange', marker='v', s=40, alpha=0.8)

ax2.set_title('BIST 100 - LSTM (Derin Öğrenme) Günlük Tahmin Dağılımı')
ax2.set_xlabel('Tarih')
ax2.set_ylabel('Endeks Puanı (TL)')
ax2.grid(True, alpha=0.3)

# Açıklama Kutuları (Legend)
yesil_ok = mlines.Line2D([], [], color='white', markerfacecolor='green', marker='^', markersize=8, label='RF Yükseliş (Al)')
kirmizi_ok = mlines.Line2D([], [], color='white', markerfacecolor='red', marker='v', markersize=8, label='RF Düşüş (Sat)')
mavi_ok = mlines.Line2D([], [], color='white', markerfacecolor='blue', marker='^', markersize=8, label='LSTM Yükseliş (Al)')
turuncu_ok = mlines.Line2D([], [], color='white', markerfacecolor='orange', marker='v', markersize=8, label='LSTM Düşüş (Sat)')

ax1.legend(handles=[yesil_ok, kirmizi_ok], loc='upper left')
ax2.legend(handles=[mavi_ok, turuncu_ok], loc='upper left')

plt.tight_layout()
plt.savefig('bist100_final_sinyal.png')
print("✅ Dengeli tahmin grafiği 'bist100_final_sinyal.png' olarak güncellendi.")
plt.show()