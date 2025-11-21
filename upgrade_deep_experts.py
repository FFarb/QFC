"""
Script to upgrade deep_experts.py for multi-asset support.
This script adds asset embeddings, sparse activation, and Monte Carlo inference.
"""

# Read the original file
with open('src/models/deep_experts.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Define the modifications
modifications = {
    # Update AdaptiveConvExpert __init__ signature
    'def __init__(\n        self,\n        n_features: int,\n        hidden_dim: int = 32,':
    'def __init__(\n        self,\n        n_features: int,\n        num_assets: int = 11,\n        embedding_dim: int = 16,\n        hidden_dim: int = 32,',
    
    # Update docstring
    '    Seven-Eye Visionary: Multi-scale convolutional network with attention.\n    \n    Architecture:\n    1. Latent Projection:':
    '    Seven-Eye Visionary: Multi-scale convolutional network with attention and asset embeddings.\n    \n    Architecture:\n    0. Asset Embedding: Learn coin-specific "personalities" (e.g., DOGE volatility vs BTC stability)\n    1. Latent Projection:',
    
    # Add parameters to docstring
    '    n_features : int\n        Number of input features (tabular data dimension).\n    hidden_dim : int, optional':
    '    n_features : int\n        Number of input features (tabular data dimension).\n    num_assets : int, optional\n        Number of unique assets for embedding layer (default: 11).\n    embedding_dim : int, optional\n        Dimension of asset embeddings (default: 16).\n    hidden_dim : int, optional',
}

# Apply modifications
for old, new in modifications.items():
    content = content.replace(old, new)

# Write back
with open('src/models/deep_experts.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✓ Updated deep_experts.py with multi-asset support")
print("  - Added num_assets and embedding_dim parameters")
print("  - Updated docstrings")
print("\nNote: Manual edits still needed for:")
print("  - Adding asset_embedding layer in __init__")
print("  - Updating forward() to accept asset_ids")
print("  - Adding predict_with_uncertainty() method")
print("  - Updating TorchSklearnWrapper")
