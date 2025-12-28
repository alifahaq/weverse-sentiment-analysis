# weverse-sentiment-analysis

Proyek ini bertujuan untuk menganalisis sentimen pengguna aplikasi Weverse di Google Play Store menggunakan algoritma **Multinomial Naive Bayes**.

## Fitur Utama
- **Crawling Data**: Mengambil 1000 ulasan terbaru menggunakan `google-play-scraper`.
- **Preprocessing**: Case Folding, Cleaning (Regex), Stopword Removal, dan Stemming (Sastrawi).
- **Modeling**: Transformasi TF-IDF dan Klasifikasi Naive Bayes.
- **Visualisasi**: WordCloud dan Distribusi Rating.

## Hasil Performa
- **Akurasi**: 91.35%

## Cara Menjalankan
1. Instal library: `pip install pandas google-play-scraper Sastrawi scikit-learn matplotlib seaborn wordcloud`
2. Jalankan file `main_script.py`.
