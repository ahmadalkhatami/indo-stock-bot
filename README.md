# 🚀 IndoStockBot v2.0 - Indonesian Stock AI Prediction System

IndoStockBot adalah sistem prediksi pasar saham Indonesia (BEI/IDX) berbasis AI yang dirancang khusus untuk trader profesional dan investor ritel. Menggunakan algoritma Machine Learning XGBoost untuk menganalisis puluhan indikator teknikal, sentimen berita, dan arus kas asing secara real-time.

## ✨ Fitur Utama
- **AI Prediction Engine**: XGBoost dengan hyperparameter tuning otomatis via Optuna.
- **Sentiment Analysis**: Scraping headline berita terbaru dari CNBC Indonesia untuk analisis sentimen pasar.
- **Strategic Configuration**: Pengaturan parameter strategi (TP/SL, Capital) terpusat di `config.yaml`.
- **Foreign Flow Intelligence**: Integrasi data arus kas asing (Net Foreign) dari Kontan Data Center.
- **Structured Logging**: Sistem pencatatan aktivitas bot yang detail di folder `logs/`.
- **Premium Dashboard**: Antarmuka berbasis Streamlit dengan desain Glassmorphism dan monitoring log real-time.
- **Telegram Notifications**: Notifikasi sinyal BUY langsung ke Telegram.

## 🛠️ Arsitektur Aplikasi
- **Backend**: Python 3.10+
- **Database**: JSON, Parquet, & SQLite (Historical Picks)
- **Model**: XGBoost (Versioned)
- **Data Source**: Yahoo Finance, Wikipedia, CNBC Indonesia, Kontan

## 📂 Struktur File Penting
- `main.py`: Pipeline utama (Fetch -> Predict -> Backtest -> Notify).
- `config.yaml`: Pusat pengaturan parameter strategi.
- `dashboard/app.py`: Dashboard analisis visual & monitoring.
- `models/versions/`: Folder histori model AI yang telah dilatih.
- `logs/`: Catatan aktivitas dan error sistem.
- `tests/`: Unit testing untuk menjaga stabilitas kode.

## 🚀 Cara Menjalankan
1.  **Persiapan Virtual Environment**:
    ```bash
    # Masuk ke environment (pastikan folder .venv dihapus jika ada duplikat)
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2.  **Konfigurasi**:
    Buka `config.yaml` untuk menyesuaikan modal awal, threshold, atau daftar saham target.
3.  **Jalankan Bot**:
    ```bash
    python3 main.py
    ```
4.  **Buka Dashboard**:
    ```bash
    streamlit run dashboard/app.py
    ```

---
Developed with ❤️ for the Indonesian Trading Community.
