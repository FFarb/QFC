"""
Deep Quant Pipeline Orchestrator.
Executes the Multi-Stage Deep Quant Pipeline:
1. Feature Engineering (Physics/Chaos)
2. Chaos Filtering (FDI)
3. Regime-Based Training (Trend vs MeanRev)
4. Walk-Forward Simulation with MAE Analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List

from src.config import (
    SYMBOL, INTERVAL, DAYS_BACK, FEATURE_STORE, 
    TP_PCT, SL_PCT, BARRIER_HORIZON
)
from src.features import build_feature_dataset
from src.models import SniperModelTrainer
from src.metrics import check_mae_mfe, profit_weighted_confusion_matrix

def run_pipeline():
    # --- 1. Data & Features ---
    print("\n=== STEP 1: Data & Physics Engine ===")
    df = build_feature_dataset(days_back=DAYS_BACK, force_refresh=True)
    
    # Ensure we have the new features
    required_cols = ["hurst_100", "fdi_100", "entropy_100"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required physics features: {required_cols}")

    # --- 2. Chaos Filter ---
    print("\n=== STEP 2: Chaos Filter (FDI) ===")
    initial_len = len(df)
    # Filter out high chaos (FDI > 1.6)
    # Note: We only filter for TRAINING. For simulation, we apply the filter dynamically.
    # Actually, the user said "Drop all rows where FDI > 1.6" as Step A.
    # Let's assume this applies to the dataset used for training/analysis.
    
    # We need to split Train/Test chronologically FIRST to avoid lookahead bias.
    train_split_idx = int(initial_len * 0.8)
    train_df = df.iloc[:train_split_idx].copy()
    test_df = df.iloc[train_split_idx:].copy()
    
    print(f"Train Set: {len(train_df)} | Test Set: {len(test_df)}")
    
    # Apply Chaos Filter to Training Data
    train_filtered = train_df[train_df["fdi_100"] <= 1.6].copy()
    filtered_count = len(train_df) - len(train_filtered)
    print(f"Filtered {filtered_count} chaotic rows ({filtered_count/len(train_df):.1%}) from Training Set.")
    
    # --- 3. Split Training (Hypothesis Injection) ---
    print("\n=== STEP 3: Regime-Based Training ===")
    
    # Trend Regime: Hurst > 0.5
    df_trend = train_filtered[train_filtered["hurst_100"] > 0.5].copy()
    print(f"Training Trend Model on {len(df_trend)} samples (Hurst > 0.5)")
    
    trainer = SniperModelTrainer()
    trend_result = trainer.run(
        input_df=df_trend, 
        model_name="Trend_Model",
        output_path="data/trend_model_data.parquet"
    )
    trend_model = trend_result["model"]
    
    # Mean Reversion Regime: Hurst <= 0.5
    df_mean_rev = train_filtered[train_filtered["hurst_100"] <= 0.5].copy()
    print(f"Training MeanRev Model on {len(df_mean_rev)} samples (Hurst <= 0.5)")
    
    mean_rev_result = trainer.run(
        input_df=df_mean_rev, 
        model_name="MeanRev_Model",
        output_path="data/meanrev_model_data.parquet"
    )
    mean_rev_model = mean_rev_result["model"]
    
    # --- 4. Walk-Forward Simulation ---
    print("\n=== STEP 4: Walk-Forward Simulation (Test Set) ===")
    
    equity = [10000.0] # Start with $10k
    equity_curve = []
    trades = []
    
    # We need to iterate through the Test Set
    # We need 'target' column for ground truth? 
    # Actually, we should simulate properly using future prices.
    # But 'target' in df is already a triple-barrier label.
    # However, for MAE analysis, we need the raw OHLCV of the *future* bars.
    # Since 'target' is pre-calculated, it doesn't give us MAE.
    # We need to look ahead.
    
    # Let's use the raw prices in test_df.
    # We need to be careful about indexing.
    
    closes = test_df["close"].values
    highs = test_df["high"].values
    lows = test_df["low"].values
    hursts = test_df["hurst_100"].values
    fdis = test_df["fdi_100"].values
    
    # Get features for prediction
    # We need the exact same feature columns used in training
    feature_cols = trend_result["top_features"] # Assuming both models use similar top features?
    # Wait, 'top_features' might differ between models!
    # We should use the specific features for each model.
    trend_features = trend_result["top_features"]
    mean_rev_features = mean_rev_result["top_features"]
    
    # Pre-select feature matrices
    X_test_trend = test_df[trend_features]
    X_test_mean_rev = test_df[mean_rev_features]
    
    # Iterate
    n_test = len(test_df)
    horizon = BARRIER_HORIZON
    
    print(f"Simulating {n_test} bars...")
    
    for i in range(n_test - horizon):
        current_equity = equity[-1]
        
        # 1. Chaos Filter
        if fdis[i] > 1.6:
            equity_curve.append(current_equity)
            continue # No Trade
            
        # 2. Regime Selection
        hurst = hursts[i]
        if hurst > 0.5:
            model = trend_model
            features = X_test_trend.iloc[i:i+1]
            regime = "Trend"
        else:
            model = mean_rev_model
            features = X_test_mean_rev.iloc[i:i+1]
            regime = "MeanRev"
            
        # 3. Prediction
        pred = model.predict(features)[0]
        
        if pred == 0:
            equity_curve.append(current_equity)
            continue # No Trade
            
        # 4. Trade Execution & MAE Check
        entry_price = closes[i]
        
        # Look ahead 'horizon' bars
        future_highs = highs[i+1 : i+1+horizon]
        future_lows = lows[i+1 : i+1+horizon]
        
        is_win, mae, mfe = check_mae_mfe(
            entry_price, future_highs, future_lows, 
            tp_pct=TP_PCT, sl_pct=SL_PCT
        )
        
        # Dynamic Leverage / Liquidation Check
        # If Leverage * MAE > 1 (100%), Liquidation.
        # Let's assume Leverage = 3 (from config)
        leverage = 3
        if mae * leverage > 1.0:
            # Liquidation!
            pnl = -1.0 # Lose 100% of margin? Or whole account?
            # Let's assume isolated margin per trade.
            # But for equity curve, let's say we lose the SL amount or Liquidation amount.
            # If liquidated, we lose the position size.
            # Let's stick to the simple PnL logic for now but mark it.
            trade_pnl_pct = -1.0 # Rekt
            outcome = "LIQUIDATION"
        elif is_win:
            trade_pnl_pct = TP_PCT * leverage
            outcome = "WIN"
        else:
            trade_pnl_pct = -SL_PCT * leverage
            outcome = "LOSS"
            
        # Update Equity (Fixed fractional or fixed size? Let's use fixed size for simplicity or compounding)
        # Compounding:
        new_equity = current_equity * (1 + trade_pnl_pct)
        equity.append(new_equity)
        
        trades.append({
            "Index": i,
            "Regime": regime,
            "Hurst": hurst,
            "FDI": fdis[i],
            "Outcome": outcome,
            "PnL": trade_pnl_pct,
            "MAE": mae,
            "MFE": mfe
        })
        
    # --- 5. Reporting ---
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        print("\n=== Simulation Results ===")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Final Equity: ${equity[-1]:.2f} (Start: $10,000)")
        print(f"Win Rate: {len(trades_df[trades_df['Outcome']=='WIN']) / len(trades_df):.2%}")
        
        # Profit-Weighted Matrix
        # We can construct it from the trades_df
        # But we need 'y_true' and 'y_pred' style.
        # Here y_pred is always 1 (since we only record trades).
        # So we just show the PnL distribution.
        
        print("\nProfit by Regime:")
        print(trades_df.groupby("Regime")["PnL"].sum())
        
        # Plot Equity Curve vs Hurst
        # We need to align equity curve with time.
        # equity_curve has length of simulation steps (minus skipped ones? No, we appended for skips).
        # Wait, we appended for skips.
        
        # Plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=equity_curve, mode='lines', name='Equity'))
        fig.update_layout(title="Deep Quant Strategy Equity Curve", template="plotly_dark")
        fig.write_html("equity_curve.html")
        print("Saved equity_curve.html")
        
        # Scatter of Trades (Hurst vs PnL)
        fig2 = px.scatter(
            trades_df, x="Hurst", y="PnL", color="Outcome", 
            title="Trade Performance vs Hurst Exponent",
            template="plotly_dark"
        )
        fig2.add_vline(x=0.5, line_dash="dash", annotation_text="Regime Boundary")
        fig2.write_html("hurst_pnl_scatter.html")
        print("Saved hurst_pnl_scatter.html")
        
    else:
        print("No trades executed.")

if __name__ == "__main__":
    run_pipeline()
