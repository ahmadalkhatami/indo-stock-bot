import os
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

PICKS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'latest_picks.csv')
THRESHOLD = 0.55


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Welcome to the Indo Stock Predictor Bot.\n"
        "Use /top to see today's high-confidence picks (probability >= 55%)."
    )


async def top_picks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not os.path.exists(PICKS_FILE):
        await update.message.reply_text(
            "No picks available yet. Please run the pipeline first."
        )
        return

    df = pd.read_csv(PICKS_FILE)
    if 'signal' in df.columns:
        df = df[df['signal'] == 'BUY']
    elif 'probability' in df.columns:
        df = df[df['probability'] >= THRESHOLD]
    else:
        df = pd.DataFrame()

    if df.empty:
        await update.message.reply_text("No high-confidence signals today.")
        return

    lines = ["*Top Picks Today* (probability >= 55%):\n"]
    for _, row in df.head(5).iterrows():
        prob = float(row.get('probability', 0)) * 100
        close = float(row.get('Close', 0))
        lines.append(f"- {row['Ticker']}: {prob:.2f}% | Rp {close:,.0f}")
    await update.message.reply_text("\n".join(lines), parse_mode='Markdown')


def run_bot(token: str):
    app = ApplicationBuilder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("top", top_picks))
    print("Telegram bot started. Press Ctrl+C to stop.")
    app.run_polling()

if __name__ == "__main__":
    bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
    if bot_token:
        print("Menjalankan Bot Standby 24/7...")
        run_bot(bot_token)
    else:
        print("Error: Mohon export TELEGRAM_BOT_TOKEN terlebih dahulu!")
