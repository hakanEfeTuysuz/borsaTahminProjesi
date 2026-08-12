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
   Bireysel hisselerin manipülatif gürültüsünden kurtulmak için BIST 100 (`XU100.IS`) endeksi baz alındı. Derin öğrenme (LSTM) ile Geleneksel Makine Öğrenmesi (Random Forest) aynı veri seti üzerinde dövüştürüldü.

## 🏆 Benchmark Sonuçları

KISS (Keep It Simple, Stupid) prensibi doğrultusunda sadece temel fiyat hareketleri, volatilite ve hareketli ortalama farkları indikatör olarak kullanılmıştır:

*   **LSTM Doğruluğu:** ~%51.4 (Karmaşık veri setinde negatif sinyalleri filtreleme zorlukları yaşadı.)
*   **Random Forest Doğruluğu:** ~%53.7 (Kaotik piyasa verisinde Karar Ağaçlarının çok daha stabil çalıştığı kanıtlandı.)

*(Not: Algo-trading dünyasında komisyonlar düşüldükten sonra %51 ve üzeri istikrarlı başarı oranları, işlem stratejileri için temel eşik olarak kabul edilir.)*

## 📊 Örnek Görselleştirme
Proje, kazanan Random Forest algoritmasının tahminlerini BIST 100 grafiği üzerine Al/Sat sinyalleri (yeşil/kırmızı oklar) olarak yerleştiren bir görselleştirme modülüne sahiptir. Modelin piyasa trendlerini yakalama kabiliyetini repodaki `bist100_final_sinyal.png` üzerinden inceleyebilirsiniz.

## 🛠️ Kullanılan Teknolojiler
*   **Python**
*   **TensorFlow & Keras** (Derin Öğrenme / LSTM)
*   **Scikit-Learn** (Random Forest, Veri Ölçeklendirme)
*   **yfinance** (Veri Tedariği)
*   **Matplotlib & Pandas** (Veri Manipülasyonu ve Görselleştirme)