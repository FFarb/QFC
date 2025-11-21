"""
Central configuration for the Quanta Futures research package.
"""

from pathlib import Path

# --- Market / data parameters -------------------------------------------------
SYMBOL = "BTCUSDT"
INTERVAL = "60"  # minutes supported by Bybit V5 API
DAYS_BACK = 90
CACHE_DIR = Path(".")
MAX_FETCH_BATCHES = 10  # Safety net for paginated API calls

# --- Strategy parameters ------------------------------------------------------
LEVERAGE = 3
TP_PCT = 0.02  # +2% take-profit
SL_PCT = 0.01  # -1% stop-loss
BARRIER_HORIZON = 36  # bars evaluated by the triple-barrier logic

# --- Dynamic Strategy Settings ---
USE_DYNAMIC_TARGETS = True  # Set to False to use static fixed %
VOLATILITY_LOOKBACK = 14    # Period for ATR calculation (if not using pre-calculated)
TP_ATR_MULT = 2.5           # Take Profit = 2.5x ATR
SL_ATR_MULT = 1.0           # Stop Loss = 1.0x ATR

# --- Modeling ----------------------------------------------------------------
FEATURE_STORE = Path("btc_1000_features.parquet")
TRAINING_SET = Path("btc_sniper_ready.parquet")
TOP_FEATURES = 25
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# --- Visualization -----------------------------------------------------------
PLOT_TEMPLATE = "plotly_dark"
