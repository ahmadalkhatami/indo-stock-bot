import schedule
import time
import subprocess
import sys
import os

def run_bot():
    print("\n[Scheduler] Menjalankan Hitungan Pipeline Harian...")
    try:
        # Hapus token khusus untuk anak proses main.py, 
        # agar loop Telegram Bot polling tidak mengunci waktu (stuck)!
        env = os.environ.copy()
        if 'TELEGRAM_BOT_TOKEN' in env:
            del env['TELEGRAM_BOT_TOKEN']
            
        subprocess.run([sys.executable, "main.py"], env=env)
        print("[Scheduler] Hitungan Harian selesai! Data terbaru tersedia untuk Bot Tele.\n")
    except Exception as e:
        print(f"[Scheduler] Terjadi Kesalahan: {e}")

# Jadwalkan setiap Senin - Jumat jam 16:30 sore (Tutup Bursa Saham Indonesia)
schedule.every().monday.at("16:30").do(run_bot)
schedule.every().tuesday.at("16:30").do(run_bot)
schedule.every().wednesday.at("16:30").do(run_bot)
schedule.every().thursday.at("16:30").do(run_bot)
schedule.every().friday.at("16:30").do(run_bot)

print("==================================================")
print("  🤖 Auto-Scheduler Bot Saham Sedang Berjalan!    ")
print("  Bot akan meminjam Terminal ini dan otomatis")
print("  mengeksekusi 'main.py' SETIAP HARI BURSA pukul 16:30 sore.")
print("  (Biarkan terminal ini terbuka/minimize ya!)")
print("==================================================")

# Loop abadi untuk mengecek detik jam
while True:
    schedule.run_pending()
    time.sleep(30) # Istirahatkan CPU selama 30 detik setiap cek
