"""
Deep Quant Pipeline Orchestrator.
Executes the "Zero-Heuristic" End-to-End ML System:
1. Feature Engineering (Physics/Chaos)
2. Universal Model Training (No Filters, No Regime Split)
3. Walk-Forward Simulation with Continuous Sizing (Kelly-like)
4. Validation (Hurst vs FDI Scatter)
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from typing import Dict, List

from src.config import (
    SYMBOL, INTERVAL, DAYS_BACK, FEATURE_STORE, 
    TP_PCT, SL_PCT, BARRIER_HORIZON, LEVERAGE
)
from src.features import build_feature_dataset
from src.models import SniperModelTrainer
from src.metrics import check_mae_mfe

def run_pipeline():
    # --- 1. Data & Features ---
    print("\n=== STEP 1: Data & Physics Engine ===")
    df = build_feature_dataset(days_back=DAYS_BACK, force_refresh=True)
    
    # Ensure we have the new features
    required_cols = ["hurst_100", "fdi_100", "entropy_100"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required physics features: {required_cols}")

    # --- 2. Universal Model Training (No Filtering) ---
    print("\n=== STEP 2: Universal Model Training ===")
    
    # Split Train/Test chronologically
    initial_len = len(df)
    train_split_idx = int(initial_len * 0.8)
    train_df = df.iloc[:train_split_idx].copy()
    test_df = df.iloc[train_split_idx:].copy()
    
    print(f"Train Set: {len(train_df)} | Test Set: {len(test_df)}")
    
    # Train Universal Model
    trainer = SniperModelTrainer()
    result = trainer.run(
        input_df=train_df, 
        model_name="Universal_Model",
        output_path="data/universal_model_data.parquet"
    )
    model = result["model"]
    top_features = result["top_features"]
    
    print(f"Top Features Selected: {top_features}")
    
    # --- 3. Walk-Forward Simulation (Continuous Sizing) ---
    print("\n=== STEP 3: Walk-Forward Simulation (Test Set) ===")
    
    equity = [10000.0] # Start with $10k
    equity_curve = []
    trades = []
    
    closes = test_df["close"].values
    highs = test_df["high"].values
    lows = test_df["low"].values
    hursts = test_df["hurst_100"].values
    fdis = test_df["fdi_100"].values
    
    # Feature Matrix for Test
    X_test = test_df[top_features]
    
    n_test = len(test_df)
    horizon = BARRIER_HORIZON
    
    print(f"Simulating {n_test} bars...")
    
    # Get probabilities for the entire test set in one go for efficiency
    # (Or loop if we want to simulate strict step-by-step, but model is static here)
    probs = model.predict_proba(X_test)[:, 1] # Probability of Class 1 (Win)
    
    for i in range(n_test - horizon):
        current_equity = equity[-1]
        prob_win = probs[i]
        
        # AI Execution Logic: Continuous Sizing
        # Confidence = Prob_Win - 0.5
        # If Prob_Win < 0.52 -> Size = 0
        
        if prob_win < 0.52:
            equity_curve.append(current_equity)
            continue # No Trade
            
        confidence = prob_win - 0.5
        scale_factor = 1.0 # Base scale
        # Position Size = Confidence * Scale_Factor * Leverage? 
        # User said: "Position_Size = Confidence * Scale_Factor"
        # Let's assume this is the % of equity to risk or allocate.
        # Let's map it to a fraction of equity.
        # E.g. Confidence 0.15 (Prob 0.65) -> Size 0.15 (15% of equity?)
        # That seems aggressive but let's follow the logic.
        # Let's cap it or use the user's LEVERAGE as a multiplier on top?
        # "Use TradePolicy only for safety limits (Max Leverage)"
        
        # Let's define Position Size as % of Equity.
        # size_pct = confidence * 2.0 (Scaling up? 0.15 * 2 = 0.3 -> 30% allocation)
        # Let's stick to: size_pct = confidence * 1.0 for now.
        # But wait, if confidence is 0.02 (0.52), size is 2%.
        
        size_pct = confidence * 2.0 # Scaling factor to make it meaningful
        
        # Safety Limit
        max_leverage = LEVERAGE
        if size_pct > max_leverage:
            size_pct = max_leverage
            
        # Calculate Position Value
        position_value = current_equity * size_pct
        
        # Trade Execution
        entry_price = closes[i]
        future_highs = highs[i+1 : i+1+horizon]
        future_lows = lows[i+1 : i+1+horizon]
        
        is_win, mae, mfe = check_mae_mfe(
            entry_price, future_highs, future_lows, 
            tp_pct=TP_PCT, sl_pct=SL_PCT
        )
        
        # PnL Calculation
        # We are Long.
        # If Win: +TP_PCT * position_value
        # If Loss: -SL_PCT * position_value
        # (Simplified, ignoring fees/slippage for now)
        
        if is_win:
            pnl = position_value * TP_PCT
            outcome = "WIN"
        else:
            pnl = -position_value * SL_PCT
            outcome = "LOSS"
            
        new_equity = current_equity + pnl
        equity.append(new_equity)
        
        trades.append({
            "Index": i,
            "Hurst": hursts[i],
            "FDI": fdis[i],
            "Prob_Win": prob_win,
            "Size_Pct": size_pct,
            "Outcome": outcome,
            "PnL": pnl,
            "MAE": mae,
            "MFE": mfe
        })
        
    # --- 4. Validation & Reporting ---
    trades_df = pd.DataFrame(trades)
    if len(trades_df) > 0:
        print("\n=== Simulation Results ===")
        print(f"Total Trades: {len(trades_df)}")
        print(f"Final Equity: ${equity[-1]:.2f} (Start: $10,000)")
        print(f"Win Rate: {len(trades_df[trades_df['Outcome']=='WIN']) / len(trades_df):.2%}")
        
        # Plot Equity Curve
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=equity_curve, mode='lines', name='Equity'))
        fig.update_layout(title="Zero-Heuristic Strategy Equity Curve", template="plotly_dark")
        fig.write_html("equity_curve.html")
        print("Saved equity_curve.html")
        
        # Scatter Plot: X=Hurst, Y=FDI, Color=Model_Probability
        # We want to see the probability landscape, not just trades.
        # So let's plot a sample of the Test Set (or all of it) with probabilities.
        # We can use the 'probs' array and the test_df features.
        
        viz_df = pd.DataFrame({
            "Hurst": hursts[:len(probs)],
            "FDI": fdis[:len(probs)],
            "Probability": probs
        })
        
        # Downsample for plotting if too large
        if len(viz_df) > 5000:
            viz_df = viz_df.sample(5000)
            
        fig2 = px.scatter(
            viz_df, x="Hurst", y="FDI", color="Probability",
            title="Model Probability Landscape (Hurst vs FDI)",
            color_continuous_scale="RdYlGn", # Red to Green
            template="plotly_dark"
        )
        # Add lines for "Chaos Corner" reference (High FDI, Low Hurst)
        fig2.add_hline(y=1.5, line_dash="dash", annotation_text="High Chaos")
        fig2.add_vline(x=0.5, line_dash="dash", annotation_text="Random Walk")
        
        fig2.write_html("physics_probability_map.html")
        print("Saved physics_probability_map.html")
        
    else:
        print("No trades executed.")

if __name__ == "__main__":
    run_pipeline()
