# 📈 Endüstriyel Zaman Serisi Analizi: LSTM ile Borsa Tahmini (Diagnostic Project)

Bu proje, Derin Öğrenme (LSTM) mimarisi kullanılarak finansal zaman serilerinin (Örn: THYAO.IS) analiz edilmesi, modelin eğitilmesi ve en önemlisi **performansının referans metriklerle (Naive Baseline) sorgulanması** amacıyla geliştirilmiş bir AR-GE dökümantasyonudur.

## 🚀 Mimarinin Temel Özellikleri

* **Güvenilir Veri Boru Hattı:** `yfinance` ile veriler çekilmiş, **Data Leakage (Veri Sızıntısı)** problemini engellemek adına Min-Max Scaler sadece eğitim setine fit edilmiştir.
* **Model Optimizasyonu:** Çift katmanlı LSTM yapısı, %20 Dropout ve **Early Stopping (Erken Durdurma)** callback'i ile ezberlemeye (Overfitting) karşı korunmuştur.
* **Değerlendirme Metrikleri:** Sadece RMSE/MAE ile yetinilmemiş; yön tahmin doğruluğu (Directional Accuracy) ve Naive Baseline karşılaştırması sisteme entegre edilmiştir.

## 📊 Mühendislik Çıkarımları ve "Gecikme Etkisi" (Lagging Effect)

Model eğitimini başarıyla tamamlayıp test verisinde 10.76 TL gibi makul bir Ortalama Mutlak Hata (MAE) yakalamış olsa da, sistemin davranışsal analizi (Bkz: `tahmin_grafigi.png`) çok kritik bir gerçeği ortaya koymuştur:

Model, borsanın kaotik yapısı içinde büyük hatalar yapmamak (Loss değerini düşük tutmak) adına **"Gecikme Etkisine" (Lagging Effect)** sığınmış ve çoğunlukla bir önceki günün kapanış fiyatını kopyalamayı öğrenmiştir. Bu durum, Naive Baseline (Yarın = Bugün) testiyle matematiksel olarak da kanıtlanmıştır.

**Gelecek Geliştirmeler (Future Work):** Bu tanı projesi göstermektedir ki; finansal piyasalarda sadece geçmiş kapanış fiyatlarını (Close) modele vermek yeterli değildir. Gelecek mimarilerde modele Hacim (Volume), RSI ve MACD gibi teknik indikatörler ekstra özellik (feature) olarak beslenecektir.

## 💻 Kurulum ve Çalıştırma

Proje, kütüphane çakışmalarını önlemek adına tamamen izole bir sanal ortamda geliştirilmiştir.

```bash
# 1. Sanal ortamı oluşturun ve aktif edin
python3 -m venv venv
source venv/bin/activate

# 2. Gerekli kütüphaneleri kurun
pip install tensorflow yfinance scikit-learn pandas numpy matplotlib

# 3. Veri çekme ve eğitim pipeline'ını başlatın
python3 data_fetch.py

Uyarı: Bu proje tamamen yazılım ve yapay zeka mimarilerini test etmek amacıyla geliştirilmiştir, finansal yatırım tavsiyesi içermez.