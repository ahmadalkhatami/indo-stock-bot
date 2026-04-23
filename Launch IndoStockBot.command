#!/bin/bash
# Launcher for Indo Stock Bot AI
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
source venv/bin/activate
python3 desktop_app.py
