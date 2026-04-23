# 📗 IndoStockBot v2.0 Official Wiki

Selamat datang di Dokumentasi Teknis IndoStockBot. Dokumen ini dirancang untuk memberikan pemahaman mendalam tentang cara kerja sistem, strategi AI yang digunakan, serta panduan operasional lengkap.

---

## 🏛️ Arsitektur Sistem

IndoStockBot bekerja dalam siklus pipeline yang terintegrasi:

1.  **Data Ingestion**: 
    - Mengambil daftar saham terupdate (LQ45/IDX80) dari Wikipedia.
    - Mengambil data harga historis (OHLCV) dari Yahoo Finance.
    - Mengambil data Foreign Flow dari Kontan Data Center.
    - Scraping headline berita terbaru dari CNBC Indonesia.
2.  **Feature Engineering**: 
    - Kalkulasi indikator teknikal (RSI, MACD, Bollinger Bands, EMA, dll).
    - Kalkulasi fitur volume dan volatilitas.
    - Analisis sentimen berita (Natural Language Processing sederhana).
    - Join data Foreign Flow dengan data harga.
3.  **Machine Learning Layer**: 
    - Model: XGBoost Classifier.
    - Tuning: Bayesian Optimization melalui library Optuna.
    - Cross-Validation: TimeSeriesSplit untuk mencegah data leakage.
4.  **Signal Generation**:
    - Model memprediksi probabilitas kenaikan harga di masa depan.
    - Sinyal BUY dihasilkan jika probabilitas melewati `confidence_threshold`.
5.  **Backtesting & Reporting**:
    - Simulasi performa berdasarkan skenario modal dan parameter di `config.yaml`.
    - Penyimpanan hasil ke `data/artifacts/` dan SQLite database.
6.  **Notification**: 
    - Pengiriman sinyal ke Telegram Bot.

---

## 🧠 Strategi AI & Model

### Fitur Prediksi (Features)
Aplikasi ini menggunakan kombinasi tiga pilar data:
- **Trend/Momentum**: EMA-cross, RSI, ADX.
- **Liquidity (Foreign Flow)**: Net Foreign Buy/Sell (Harian & Akumulasi 5 hari).
- **Sentiment**: Skor berita positif/negatif dari media finansial utama.

### Model Versioning
Setiap kali `main.py` selesai melatih model, file model `.pkl` disimpan di `models/versions/` dengan timestamp. Hal ini memungkinkan pengguna untuk melakukan *rollback* atau membandingkan performa antar versi model.

---

## ⚙️ Panduan Konfigurasi (`config.yaml`)

File ini adalah jantung dari strategi trading Anda:

- `initial_capital`: Modal awal dalam IDR (default: 100jt).
- `risk_per_trade`: Persentase modal per satu saham.
- `take_profit_pct`: Target keuntungan (misal: 0.05 untuk 5%).
- `stop_loss_pct`: Batas kerugian (misal: -0.03 untuk -3%).
- `confidence_threshold`: Ambang batas probabilitas AI (misal: 0.70 untuk 70% keyakinan).

---

## 📋 Operasional Harian

### Menggunakan UI Desktop
Jalankan file `Launch IndoStockBot.command`.
1.  **Overview**: Lihat ringkasan performa portofolio metrik (Return, Win Rate, Alpha).
2.  **Stock Chart Explorer**: Fitur untuk melihat pergerakan harga riil (live candlestick) dari saham-saham pilihan AI secara interaktif dengan indikator EMA.
3.  **Top Predictions**: Hasil analisa harian (saham yang masuk rating BUY).
4.  **Trading Logbook**: Buku jurnal interaktif untuk mencatat setiap posisi beli/jual secara permanen.
5.  **Scheduler**: Atur jam eksekusi otomatis (biasanya jam 16:30 WIB saat pasar tutup).
6.  **Manual Execution**: Klik "🚀 EXECUTE AI PIPELINE" jika ingin menjalankan analisis saat ini juga.
7.  **System Logs**: Gunakan tab ini untuk melihat diagnostic.

### Troubleshooting
- **Missing Data**: Jika data yfinance gagal, periksa koneksi internet atau coba jalankan kembali pipeline.
- **Telegram Tidak Kirim Sinyal**: Pastikan `TELEGRAM_BOT_TOKEN` sudah diset di environment variable atau hubungi bot Anda untuk memastikan bot aktif.
- **Model Error**: Jika model gagal dimuat, hapus isi `models/versions/` dan jalankan `main.py` untuk melatih ulang model baru.

---

## 🛠️ Pengembangan (Developer Notes)

- **Library Utama**: `pandas`, `xgboost`, `streamlit`, `yfinance`, `beautifulsoup4`.
- **Unit Testing**: Jalankan `pytest tests/` untuk memverifikasi logika fitur.
- **Logging**: Semua log teknis tercatat di `logs/bot_YYYYMMDD.log`.

---
*Created and Maintained by AI Professional Upgrade Service.*
