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
from .features import SignalFactory, add_signal_interactions, build_feature_dataset
from .models import (
    get_triple_barrier_labels,
    SniperModelTrainer,
    filter_correlated_features,
)
from .meta_model import MetaModelTrainer

__all__ = [
    "MarketDataLoader",
    "SignalFactory",
    "build_feature_dataset",
    "add_signal_interactions",
    "visualize_data",
    "get_triple_barrier_labels",
    "filter_correlated_features",
    "SniperModelTrainer",
    "MetaModelTrainer",
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
