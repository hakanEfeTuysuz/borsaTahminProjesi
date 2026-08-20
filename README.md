# 📈 Yapay Zeka ile BIST 100 Yön Tahmin Analizi (Benchmark Projesi)

Bu proje, makine öğrenmesi ve derin öğrenme algoritmalarının finansal zaman serilerindeki (Borsa İstanbul) davranışlarını analiz etmek, hatalı varsayımları ayıklamak ve en stabil tahminciyi bulmak amacıyla geliştirilmiş bir veri bilimi AR-GE çalışmasıdır.

## 🚀 Projenin Hikayesi ve Gelişim Süreci

Bu projede doğrudan "Mükemmel" bir model kurmak yerine, gerçek dünya verilerinin getirdiği zorluklarla yüzleşilerek iteratif bir mühendislik yaklaşımı benimsenmiştir:

1. **Regresyon Tuzağı (İlk Deneme):**
   İlk olarak LSTM modeli ile "Hisse yarın tam olarak yüzde kaç artacak?" sorusuna (Regresyon) cevap arandı. Ancak model, karesel hata (MSE) cezalarından kaçınmak için uçuk tahminler yapmayı bırakıp sürekli "0" (ortalama) değerine yaklaşarak korkak bir tutum sergiledi.
2. **Veri Açlığı ve Toprak Uyuşmazlığı:**
   Algoritma yeni halka arz olmuş, dar bir veri setine sahip hisselerde denendiğinde, derin öğrenme (LSTM) modellerinin verisizlikten dolayı "Dying ReLU" ve ezberleme krizlerine girdiği gözlemlendi.
3. **Sınıflandırmaya (Classification) Geçiş:**
   Fiyat tahmini yerine, problem ikili sınıflandırmaya (Yükselecek: 1 / Düşecek: 0) dönüştürüldü.
4. **Endeks Verisi ve Algoritma Çarpışması:**
   Bireysel hisselerin manipülatif gürültüsünden kurtulmak için BIST 100 (`XU100.IS`) endeksi baz alındı.
5. **Zaman Serisi Doğrulaması ve İleri Özellik Mühendisliği (V3):**
   Veri sızıntısı (data leakage) riski; eğitim, doğrulama ve test setlerinin kronolojik olarak (%70, %15, %15) kesin çizgilerle ayrılmasıyla tamamen giderildi. `MinMaxScaler`, sadece eğitim setine `fit` edilip tüm veriye uygulanarak geleceğe ait bilginin geçmişe sızması engellendi. Modelin piyasa bağlamını anlayabilmesi için Return, Momentum, ROC, MA farkı, RSI, MACD (+ sinyal hattı), Bollinger Bant pozisyonu, ATR ve Hacim oranı olmak üzere **10 kritik indikatör** sisteme dahil edildi. LSTM ve Random Forest algoritmalarına eşit zaman pencereleri (15 gün x 10 özellik = 150 boyut) sunularak adil bir benchmarking ortamı kuruldu.

## 🏆 V3 Benchmark Sonuçları

Geliştirilen nihai V3 mimarisiyle, modellerin sadece "sürekli yükselecek" deme eğilimi (Baseline) engellenmiş ve piyasanın karmaşık gürültüsü içinde gerçek bir matematiksel sinyal (edge) aranmıştır:

| Model | Accuracy | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Baseline (Çoğunluk Sınıfı) | %51.6 | — | — | — | — |
| Random Forest | %51.8 | 0.541 | 0.528 | 0.669 | 0.590 |
| **LSTM (Şampiyon)** | **%56.3** | **0.568** | **0.585** | 0.543 | 0.563 |

*(Not: Algo-trading dünyasında komisyonlar düşüldükten sonra %51 ve üzeri istikrarlı başarı oranları, işlem stratejileri için temel eşik olarak kabul edilir. Random Forest yüksek Recall'a sahip olsa da düşük Precision ile fazla "yanlış alarm" üretmektedir; LSTM ise Precision ve Recall arasında daha dengeli bir denge kurarak modelin BIST 100 gürültüsünü filtrelemede görece daha güvenilir olduğunu göstermektedir. Bu sonuçlar, kısa vadeli piyasa yönü tahmininin doğası gereği zor bir problem olduğunu; modelin rastgeleden (%50) az bir farkla iyi olduğunu ve tek başına bir yatırım stratejisi olarak kullanılmaması gerektiğini de açıkça ortaya koymaktadır.)*

## 📊 Örnek Görselleştirme
Proje, `matplotlib` kullanarak modelin tahminlerini BIST 100 fiyat grafiği üzerine Al/Sat sinyalleri (yükseliş: yeşil/mavi ok yukarı, düşüş: kırmızı/turuncu ok aşağı) olarak yerleştiren, Random Forest ve LSTM'i yan yana karşılaştıran bir görselleştirme modülüne sahiptir. Çıktı `bist100_final_sinyal.png` olarak kaydedilir; algoritmaların piyasa trendlerini yakalama kabiliyetini repodaki grafik üzerinden inceleyebilirsiniz.

## ⚙️ Metodoloji Özeti
* **Veri kaynağı:** `yfinance` üzerinden 2020'den günümüze BIST 100 (`XU100.IS`) günlük OHLCV verisi.
* **Hedef değişken:** Bir sonraki günün kapanışı bugünkünden yüksekse `1`, değilse `0`.
* **Zaman penceresi:** Son 15 günlük veri, sonraki günün yönünü tahmin etmek için modele sunulur.
* **Bölünme:** Kronolojik %70 eğitim / %15 doğrulama / %15 test (karıştırma yok — geleceğe sızıntı önlenir).
* **Random Forest:** 300 ağaç, `max_depth=8`, `min_samples_leaf=5`.
* **LSTM:** 32 birimli LSTM katmanı + Dropout + Dense katmanları, `EarlyStopping` (patience=5) ile aşırı öğrenmeye karşı korunur.
* **Tekrarlanabilirlik:** `numpy`, `tensorflow` ve `random` için sabit `seed=42`.

## 🛠️ Kullanılan Teknolojiler
* **Python**
* **TensorFlow & Keras** (Derin Öğrenme / LSTM)
* **Scikit-Learn** (Random Forest, Veri Ölçeklendirme, Gelişmiş Metrikler)
* **yfinance** (Veri Tedariği)
* **NumPy & Pandas** (Veri Manipülasyonu)
* **Matplotlib** (Görselleştirme)

## ▶️ Çalıştırma
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 data_fetch.py
```

## ⚠️ Sorumluluk Reddi
Bu proje eğitim ve araştırma amaçlıdır, bir yatırım tavsiyesi değildir. Modelin ürettiği sinyaller gerçek bir alım-satım stratejisinin temeli olarak kullanılmamalıdır.