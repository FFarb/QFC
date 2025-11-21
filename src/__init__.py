"""
Quanta Futures
==============

High-level imports for the refactored package.
"""

from .config import (
    SYMBOL,
    INTERVAL,
    DAYS_BACK,
    CACHE_DIR,
    LEVERAGE,
    TP_PCT,
    SL_PCT,
    BARRIER_HORIZON,
    FEATURE_STORE,
    TRAINING_SET,
)
from .data_loader import MarketDataLoader, visualize_data
from .features import SignalFactory, build_feature_dataset
from .models import (
    get_triple_barrier_labels,
    SniperModelTrainer,
    filter_correlated_features,
)

__all__ = [
    "MarketDataLoader",
    "SignalFactory",
    "build_feature_dataset",
    "visualize_data",
    "get_triple_barrier_labels",
    "filter_correlated_features",
    "SniperModelTrainer",
    "SYMBOL",
    "INTERVAL",
    "DAYS_BACK",
    "CACHE_DIR",
    "LEVERAGE",
    "TP_PCT",
    "SL_PCT",
    "BARRIER_HORIZON",
    "FEATURE_STORE",
    "TRAINING_SET",
]
