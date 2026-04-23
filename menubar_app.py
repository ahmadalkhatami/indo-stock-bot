import rumps
import subprocess
import os
import threading
import sys
import time

class IndoStockMenuApp(rumps.App):
    def __init__(self):
        # Icon menu bar 📈
        super(IndoStockMenuApp, self).__init__("📈 AI Stock")
        self.menu = ["Mulai Prediksi AI", "Buka Dashboard", rumps.separator, "Keluar"]
        
    @rumps.clicked("Mulai Prediksi AI")
    def run_prediction(self, _):
        rumps.notification("Indo Stock Bot", "Memulai Analisis AI...", "Mohon tunggu ~2 menit.")
        def run_it():
            try:
                env = os.environ.copy()
                if 'TELEGRAM_BOT_TOKEN' in env:
                    del env['TELEGRAM_BOT_TOKEN']
                    
                base_dir = os.path.dirname(os.path.abspath(__file__))
                python_exe = os.path.join(base_dir, "venv", "bin", "python3")
                main_script = os.path.join(base_dir, "main.py")
                
                subprocess.run([python_exe, main_script], env=env)
                rumps.notification("Indo Stock Bot", "Analisis Selesai!", "Data terbaru sudah masuk ke Dashboard.")
            except Exception as e:
                rumps.notification("Indo Stock Bot", "Gagal!", f"Error: {str(e)}")
                
        threading.Thread(target=run_it, daemon=True).start()

    @rumps.clicked("Buka Dashboard")
    def open_dash(self, _):
        def open_it():
            import webbrowser
            base_dir = os.path.dirname(os.path.abspath(__file__))
            streamlit_exe = os.path.join(base_dir, "venv", "bin", "streamlit")
            dashboard_script = os.path.join(base_dir, "dashboard", "app.py")
            subprocess.Popen([streamlit_exe, "run", dashboard_script, "--server.headless=true"])
            time.sleep(2)
            webbrowser.open("http://localhost:8501")
        
        threading.Thread(target=open_it, daemon=True).start()

if __name__ == "__main__":
    IndoStockMenuApp().run()
