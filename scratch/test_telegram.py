import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

from utils.notifications import send_telegram_signal

TOKEN = "8633863682:AAGXhPc8RSPsmm2sY3xtWedhhDukkuL6vMY"
CHAT_ID = "1569995460"

test_msg = (
    "✅ *Koneksi Berhasil!*\n\n"
    "Halo Ahmad! Ini adalah bot sinyal saham pribadi kamu.\n"
    "Sistem sudah terhubung 100% dan siap mengirimkan cuan ke HP kamu. 🚀📈"
)

print("Memulai tes pengiriman pesan ke Telegram...")
success = send_telegram_signal(TOKEN, CHAT_ID, test_msg)

if success:
    print("TES SUKSES! Silakan cek Telegram kamu.")
else:
    print("TES GAGAL. Periksa kembali Token atau ID kamu.")
