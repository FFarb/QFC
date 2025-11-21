"""
Deep Quant Pipeline Orchestrator (Smart Cache & Optimized Memory Edition).
Multi-Asset Fix: Decoupled Feature Generation & Disk Offloading.
"""
from __future__ import annotations

import gc
import shutil
from pathlib import Path
from typing import List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

# Import Config
from src.config import DAYS_BACK, SYMBOLS, CACHE_DIR
ASSET_LIST = SYMBOLS

from src.features import SignalFactory
from src.features.advanced_stats import apply_rolling_physics
from src.features.alpha_council import AlphaCouncil
from src.models.moe_ensemble import MixtureOfExpertsEnsemble
from src.training.meta_controller import TrainingScheduler
from src.data_loader import MarketDataLoader
from src.analysis.threshold_tuner import run_tuning

PHYSICS_COLUMNS: Sequence[str] = ("hurst_200", "entropy_200", "fdi_200")
LABEL_LOOKAHEAD = 36
LABEL_THRESHOLD = 0.005
TEMP_DIR = Path("temp_processed_assets")

def _validate_physics_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        pass 

def _build_labels(df: pd.DataFrame) -> pd.Series:
    if 'asset_id' in df.columns:
        forward_ret = df.groupby('asset_id')['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    else:
        forward_ret = df['close'].shift(-LABEL_LOOKAHEAD) / df['close'] - 1.0
    y = (forward_ret > LABEL_THRESHOLD).astype(int)
    return y

def cleanup_temp_dir():
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR)
    TEMP_DIR.mkdir(exist_ok=True)

def get_smart_data(loader: MarketDataLoader, symbol: str, interval: str, days_back: int) -> pd.DataFrame:
    """
    Smart fetcher: Checks if data is up-to-date in cache before hitting API.
    Leverages the MarketDataLoader's internal caching logic but ensures we don't
    force re-download if we have the files.
    """
    # Set loader params
    loader.symbol = symbol
    loader.interval = interval
    
    # The loader.get_data() method already handles caching logic (checking parquet).
    # We just need to ensure we don't force a refresh unless necessary.
    # Assuming get_data(days_back=...) handles the 'update tail' logic internally.
    try:
        df = loader.get_data(days_back=days_back)
        return df
    except Exception as e:
        print(f"       [WARNING] Could not fetch data for {symbol} {interval}: {e}")
        return pd.DataFrame()

def run_pipeline() -> None:
    print("=" * 72)
    print("          MULTI-ASSET NEURO-SYMBOLIC TRADING SYSTEM")
    print("          (Smart Cache & Disk-Optimized Mode)")
    print("=" * 72)

    # ------------------------------------------------------------------ #
    # 1. Multi-Asset Data & Feature Factory (Iterative Disk Offload)
    # ------------------------------------------------------------------ #
    print("\n[1] DATA & FEATURE FACTORY")
    
    cleanup_temp_dir()
    
    loader = MarketDataLoader(interval="5")
    factory = SignalFactory()
    
    print(f"    Processing {len(ASSET_LIST)} assets: {ASSET_LIST}")
    
    generated_files = []
    
    for asset_idx, symbol in enumerate(ASSET_LIST):
        try:
            print(f"\n    >> Processing {symbol} (ID: {asset_idx})...")
            
            # --- A. Smart Fetch ---
            # 1. M5 Data
            df_m5 = get_smart_data(loader, symbol, "5", DAYS_BACK)
            if df_m5.empty: continue

            # 2. H1 Data
            df_h1 = get_smart_data(loader, symbol, "60", DAYS_BACK)
            if df_h1.empty: continue
            
            # --- B. Decoupled Feature Generation ---
            
            # 1. Generate Strategic Features (H1)
            print(f"       Generating Strategic (H1) features...")
            df_h1_features = factory.generate_signals(df_h1)
            df_h1_features = df_h1_features.add_prefix("macro_")
            
            # 2. Generate Execution Features (M5) - ON RAW DATA
            print(f"       Generating Execution (M5) features...")
            df_m5_features = factory.generate_signals(df_m5)
            
            # --- C. Fractal Merge ---
            print(f"       Merging Contexts...")
            df_m5_features = df_m5_features.sort_index()
            df_h1_features = df_h1_features.sort_index()
            
            df_merged = pd.merge_asof(
                df_m5_features,
                df_h1_features,
                left_index=True,
                right_index=True,
                direction='backward'
            )
            
            # --- D. Physics & Metadata ---
            print(f"       Applying Chaos Physics...")
            df_merged = apply_rolling_physics(df_merged, windows=[100, 200])
            df_merged['asset_id'] = asset_idx
            
            # --- E. Optimization ---
            df_merged = df_merged.replace([np.inf, -np.inf], np.nan).dropna()
            cols = df_merged.select_dtypes('float64').columns
            df_merged[cols] = df_merged[cols].astype('float32')
            
            # --- F. Offload ---
            save_path = TEMP_DIR / f"{symbol}.parquet"
            df_merged.to_parquet(save_path, compression='snappy')
            generated_files.append(save_path)
            print(f"       -> Saved: {save_path} | Shape: {df_merged.shape}")
            
            # --- G. Cleanup ---
            del df_m5, df_h1, df_h1_features, df_m5_features, df_merged
            gc.collect()
            
        except Exception as e:
            print(f"       [ERROR] Failed {symbol}: {e}")
            continue
    
    if not generated_files:
        print("CRITICAL: No data generated.")
        return
    
    # ------------------------------------------------------------------ #
    # 2. Global Assembly
    # ------------------------------------------------------------------ #
    print("\n[2] GLOBAL TENSOR ASSEMBLY")
    try:
        df = pd.concat([pd.read_parquet(f) for f in generated_files]).sort_index()
        print(f"    Global Dataset Shape: {df.shape}")
    except MemoryError:
        print("    [CRITICAL] Memory Full. Switching to Sub-sampling Mode.")
        dfs = []
        for f in generated_files:
            d = pd.read_parquet(f)
            dfs.append(d.iloc[-int(len(d)*0.5):]) # Take last 50%
        df = pd.concat(dfs).sort_index()
        print(f"    Sampled Shape: {df.shape}")

    # ------------------------------------------------------------------ #
    # 3. Alpha Council
    # ------------------------------------------------------------------ #
    print("\n[3] ALPHA COUNCIL")
    raw_labels = _build_labels(df)
    valid_mask = ~raw_labels.isna()
    df_clean = df.loc[valid_mask]
    y_clean = raw_labels.loc[valid_mask]
    
    if len(df_clean) > 200000:
        print(f"    Downsampling for Feature Selection...")
        df_council = df_clean.iloc[-100000:]
        y_council = y_clean.iloc[-100000:]
    else:
        df_council, y_council = df_clean, y_clean

    exclude_cols = {"open", "high", "low", "close", "volume", "timestamp", "target", "asset_id", *PHYSICS_COLUMNS}
    candidates = [col for col in df_clean.columns if col not in exclude_cols]
    
    council = AlphaCouncil()
    survivors = council.screen_features(df_council[candidates], y_council)
    print(f"    Selected {len(survivors)} features.")

    available_physics = [c for c in PHYSICS_COLUMNS if c in df_clean.columns]
    final_features = survivors + available_physics + ['asset_id']
    
    X = df_clean[final_features]
    y = y_clean

    del df, df_clean, df_council, y_clean
    gc.collect()

    # ------------------------------------------------------------------ #
    # 4. Training
    # ------------------------------------------------------------------ #
    print("\n[4] MIXED MODE TRAINING")
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    scheduler = TrainingScheduler()
    # Safe access to entropy/volatility
    e_col = "entropy_200" if "entropy_200" in X.columns else X.columns[0]
    v_col = "fdi_200" if "fdi_200" in X.columns else X.columns[0]
    
    entropy_signal = float(X_train[e_col].iloc[-1000:].mean())
    volatility_signal = float(X_train[v_col].iloc[-1000:].mean())
    
    depth = scheduler.suggest_training_depth(entropy_signal, max(volatility_signal, 1e-6))
    print(f"    Config: {depth}")

    moe = MixtureOfExpertsEnsemble(
        physics_features=available_physics,
        random_state=42,
        trend_estimators=depth["n_estimators"],
        gating_epochs=depth["epochs"],
    )
    moe.fit(X_train, y_train)

    # ------------------------------------------------------------------ #
    # 5. Validation
    # ------------------------------------------------------------------ #
    print("\n[5] VALIDATION")
    probs = moe.predict_proba(X_test)[:, 1]
    
    artifacts = Path("artifacts")
    artifacts.mkdir(exist_ok=True)
    
    val_df = pd.DataFrame({"probability": probs, "target": y_test.values})
    val_df.to_parquet(artifacts / "money_machine_snapshot.parquet")
    
    run_tuning()
    cleanup_temp_dir()

if __name__ == "__main__":
    run_pipeline()
