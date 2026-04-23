import webview
import subprocess
import threading
import time
import os
import sys
import schedule

# --- CONFIGURATION ---
ROOT = "/Users/ahmadalkhatami/Documents/Stock-Bot/indo-stock-bot"
STREAMLIT = os.path.join(ROOT, "venv", "bin", "streamlit")
APP_PY = os.path.join(ROOT, "dashboard", "app.py")
PYTHON = os.path.join(ROOT, "venv", "bin", "python3")
SCHED_TIME = "16:30"

class IndoStockSuite:
    def __init__(self):
        self.server_process = None
        self.scheduler_running = True

    def start_streamlit(self):
        """Menjalankan engine dashboard di latar belakang."""
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        self.server_process = subprocess.Popen(
            [STREAMLIT, "run", APP_PY, "--server.headless=true", "--server.port=8501"],
            cwd=ROOT, env=env
        )

    def run_ai_task(self):
        """Tugas utama yang dijalankan oleh scheduler."""
        print(f"[{time.ctime()}] Scheduler triggering AI Prediction...")
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
        subprocess.Popen([PYTHON, "main.py"], cwd=ROOT, env=env)

    def scheduler_loop(self):
        """Loop penjadwalan otomatis yang dinamis."""
        import json
        config_path = os.path.join(ROOT, "data", "scheduler_config.json")
        current_time = ""
        current_active = False

        while self.scheduler_running:
            try:
                if os.path.exists(config_path):
                    with open(config_path, "r") as f:
                        config = json.load(f)
                    
                    # Update jadwal jika ada perubahan
                    if config["time"] != current_time or config["active"] != current_active:
                        schedule.clear()
                        if config["active"]:
                            t = config["time"]
                            for day in [schedule.every().monday, schedule.every().tuesday, 
                                        schedule.every().wednesday, schedule.every().thursday, 
                                        schedule.every().friday]:
                                day.at(t).do(self.run_ai_task)
                            print(f"[{time.ctime()}] Scheduler UPDATED to {t}")
                        
                        current_time = config["time"]
                        current_active = config["active"]
            except Exception as e:
                print(f"Scheduler Error: {e}")

            schedule.run_pending()
            time.sleep(5)

    def on_closed(self):
        """Membersihkan proses saat aplikasi ditutup."""
        self.scheduler_running = False
        if self.server_process:
            self.server_process.terminate()
        os._exit(0)

def run_app():
    suite = IndoStockSuite()
    
    # 1. Jalankan Dashboard Server
    threading.Thread(target=suite.start_streamlit, daemon=True).start()
    
    # 2. Jalankan Scheduler
    threading.Thread(target=suite.scheduler_loop, daemon=True).start()
    
    # Tunggu sebentar agar server siap
    time.sleep(4)
    
    # 3. Buat Jendela Desktop UI
    window = webview.create_window(
        'Indo Stock Bot - Command Center', 
        'http://localhost:8501',
        width=1100, 
        height=800,
        resizable=True,
        confirm_close=True,
        background_color='#0d1117'
    )
    
    window.events.closed += suite.on_closed
    webview.start()

if __name__ == "__main__":
    run_app()
