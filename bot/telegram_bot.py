import os
import pandas as pd
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

PICKS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'latest_picks.csv')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handler for /start command."""
    await update.message.reply_text(
        "Hello! I'm the Indo Stock Predictor Bot.\n"
        "Use /top picks to get today's predictions."
    )

async def top_picks(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handler for /top command. 
    It checks if the user specifically asked for 'picks' as an argument.
    """
    # Check if the user passed 'picks' as an argument, e.g. "/top picks"
    if not context.args or context.args[0].lower() != "picks":
        await update.message.reply_text("Please use `/top picks` to get the latest recommendations.", parse_mode='Markdown')
        return

    try:
        if not os.path.exists(PICKS_FILE):
            await update.message.reply_text("Picks are not available yet. Please run the pipeline first.")
            return
            
        df = pd.read_csv(PICKS_FILE)
        if df.empty:
            await update.message.reply_text("No picks generated for today.")
            return
            
        message = "📈 **Top 5 Indonesian Stock Picks** 📈\n\n"
        for idx, row in df.iterrows():
            message += f"{idx+1}. {row['Ticker']} - {row['Probability']*100:.2f}% probability\n"
            message += f"   Last Close: Rp {row['Close']:,.0f}\n\n"
            
        await update.message.reply_text(message, parse_mode='Markdown')
    except Exception as e:
        await update.message.reply_text(f"Error fetching picks: {e}")

def run_bot(token: str):
    """
    Start the Telegram bot.
    """
    print("Starting Telegram Bot...")
    app = ApplicationBuilder().token(token).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("top", top_picks))
    
    app.run_polling()
