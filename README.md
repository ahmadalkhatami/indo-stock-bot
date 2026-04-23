# 🚀 IndoStockBot v2.0 - Indonesian Stock AI Prediction System

IndoStockBot adalah sistem prediksi pasar saham Indonesia (BEI/IDX) berbasis AI yang dirancang khusus untuk trader profesional dan investor ritel. Menggunakan algoritma Machine Learning XGBoost untuk menganalisis puluhan indikator teknikal, sentimen global, dan arus kas asing secara real-time.

## ✨ Fitur Utama
- **AI Prediction Engine**: Menggunakan XGBoost dengan hyperparameter tuning otomatis (Optuna).
- **Anti-Block Data Loader**: Mengambil daftar saham (LQ45/IDX80) dari Wikipedia untuk stabilitas 100%.
- **Foreign Flow Intelligence**: Integrasi data arus kas asing (Net Foreign) dari Kontan Data Center.
- **Global Macro Analysis**: Memantau Kurs USD/IDR dan S&P 500 untuk akurasi prediksi makro.
- **Native macOS Desktop Interface**: Jendela aplikasi elegan berbasis WebView khusus untuk pengguna Mac.
- **Auto Scheduler**: Eksekusi prediksi otomatis setiap hari pada jam penutupan bursa.
- **Telegram Notifications**: Mengirimkan sinyal BUY/SELL langsung ke ponsel Anda.

## 🛠️ Arsitektur Aplikasi
- **Backend Core**: Python 3.10+
- **Database/Storage**: JSON Caching & Parquet
- **Dashboard**: Streamlit (Control Center)
- **GUI Framework**: PyWebView
- **Machine Learning**: XGBoost, Scikit-Learn
- **Data Source**: Yahoo Finance, Wikipedia, Kontan

## 📦 File Penting
- `desktop_app.py`: Titik masuk utama aplikasi desktop.
- `main.py`: Mesin utama AI untuk pelatihan dan prediksi.
- `dashboard/app.py`: Antarmuka visual (Dashboard).
- `data/data_loader.py`: Modul pengambilan data cerdas.
- `features/feature_engineering.py`: Modul pengolahan data teknis.

## 🚀 Cara Menjalankan
1. Pastikan Anda berada di virtual environment: `source venv/bin/activate`
2. Jalankan aplikasi desktop: `python3 desktop_app.py`
3. Atau buka dashboard saja: `streamlit run dashboard/app.py`

---
Developed with ❤️ for the Indonesian Trading Community.
