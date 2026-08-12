# 📈 Yapay Zeka ile BIST 100 Yön Tahmin Analizi (Benchmark Projesi)

Bu proje, makine öğrenmesi ve derin öğrenme algoritmalarının finansal zaman serilerindeki (Borsa İstanbul) davranışlarını analiz etmek, hatalı varsayımları ayıklamak ve en stabil tahminciyi bulmak amacıyla geliştirilmiş bir veri bilimi AR-GE çalışmasıdır.

## 🚀 Projenin Hikayesi ve Gelişim Süreci

Bu projede doğrudan "mükemmel" bir model kurmak yerine, gerçek dünya verilerinin getirdiği zorluklarla yüzleşilerek iteratif bir mühendislik yaklaşımı benimsenmiştir:

1. **Regresyon Tuzağı (İlk Deneme):** 
   İlk olarak LSTM modeli ile "Hisse yarın tam olarak yüzde kaç artacak?" sorusuna (Regresyon) cevap arandı. Ancak model, karesel hata (MSE) cezalarından kaçınmak için uçuk tahminler yapmayı bırakıp sürekli "0" (ortalama) değerine yaklaşarak korkak bir tutum sergiledi.
2. **Veri Açlığı ve Toprak Uyuşmazlığı:** 
   Algoritma yeni halka arz olmuş, dar bir veri setine sahip hisselerde denendiğinde, derin öğrenme (LSTM) modellerinin verisizlikten dolayı "Dying ReLU" ve ezberleme krizlerine girdiği gözlemlendi.
3. **Sınıflandırmaya (Classification) Geçiş:** 
   Fiyat tahmini yerine, problem ikili sınıflandırmaya (Yükselecek: 1 / Düşecek: 0) dönüştürüldü.
4. **Endeks Verisi ve Algoritma Çarpışması:** 
   Bireysel hisselerin manipülatif gürültüsünden kurtulmak için BIST 100 (`XU100.IS`) endeksi baz alındı. 
5. **Zaman Serisi Doğrulaması ve İleri Özellik Mühendisliği (V2):** 
   Başlangıçtaki veri sızıntısı (data leakage) riskleri; eğitim, doğrulama ve test setlerinin kronolojik olarak (%70, %15, %15) kesin çizgilerle ayrılmasıyla tamamen giderildi. Modelin piyasa bağlamını anlayabilmesi için RSI, MACD, ATR, Momentum ve Bollinger Bantları gibi 10 kritik indikatör sisteme dahil edildi. LSTM ve Random Forest algoritmalarına eşit zaman pencereleri (15 gün x 10 özellik = 150 boyut) sunularak adil bir benchmarking ortamı kuruldu.

## 🏆 V2 Benchmark Sonuçları

Geliştirilen nihai V2 mimarisiyle, modellerin sadece "sürekli yükselecek" deme eğilimi (Baseline) engellenmiş ve piyasanın karmaşık gürültüsü içinde gerçek bir matematiksel sinyal (edge) aranmıştır:

*   **Baseline (Çoğunluk Sınıfı Tahmini):** %51.6 Doğruluk
*   **Random Forest:** Accuracy: %51.8 | AUC: 0.541 | Precision: 0.528
*   **LSTM (Şampiyon):** Accuracy: %56.3 | AUC: 0.568 | Precision: 0.585

*(Not: Algo-trading dünyasında komisyonlar düşüldükten sonra %51 ve üzeri istikrarlı başarı oranları, işlem stratejileri için temel eşik olarak kabul edilir. LSTM'nin %58.5'lik Precision skoru, modelin BIST 100 gürültüsünü filtrelemeyi başardığını ve "Yükselecek" dediği anlarda sağladığı yüksek güvenilirliği kanıtlamaktadır.)*

## 📊 Örnek Görselleştirme
Proje, modelin tahminlerini BIST 100 grafiği üzerine Al/Sat sinyalleri (yeşil/kırmızı oklar) olarak yerleştiren bir görselleştirme modülüne sahiptir. Algoritmaların piyasa trendlerini yakalama kabiliyetini repodaki grafikler üzerinden inceleyebilirsiniz.

## 🛠️ Kullanılan Teknolojiler
*   **Python**
*   **TensorFlow & Keras** (Derin Öğrenme / LSTM)
*   **Scikit-Learn** (Random Forest, Veri Ölçeklendirme, Gelişmiş Metrikler)
*   **yfinance** (Veri Tedariği)
*   **Matplotlib & Pandas** (Veri Manipülasyonu ve Görselleştirme)