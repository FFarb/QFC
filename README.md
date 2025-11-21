<<<<<<< HEAD
# Bybit Futures Data Fetcher & Visualizer

A professional Python tool for fetching and visualizing USDT Perpetual Futures data from Bybit with interactive charts.

## 📦 Installation

Install the required dependencies:

```bash
pip install pybit pandas plotly pyarrow
```

## 🚀 Quick Start

### Option 1: Run Demo (Offline)

Test the visualization with synthetic data:

```bash
python demo_visualization.py
```

This generates realistic BTC price data and creates an interactive HTML chart.

### Option 2: Fetch Live Data

Fetch real data from Bybit:

```bash
python bybit_futures_fetcher.py
```

**Note**: Requires non-US IP address or VPN due to Bybit's geographic restrictions.

## 📊 Usage Examples

### Basic Usage

```python
from bybit_futures_fetcher import BybitFuturesLoader, visualize_data
from datetime import datetime, timedelta

# Initialize
loader = BybitFuturesLoader()

# Fetch 3 months of hourly BTCUSDT data
end_date = datetime.now()
start_date = end_date - timedelta(days=90)

df = loader.fetch_history(
    symbol="BTCUSDT",
    interval="60",  # 1 hour
    start_date=start_date,
    end_date=end_date
)

# Save to parquet
loader.save_to_parquet(df, "btc_futures.parquet")

# Visualize
visualize_data(df, "BTCUSDT")
```

### Supported Intervals

- `"1"` - 1 minute
- `"3"` - 3 minutes
- `"5"` - 5 minutes
- `"15"` - 15 minutes
- `"30"` - 30 minutes
- `"60"` - 1 hour (default)
- `"120"` - 2 hours
- `"240"` - 4 hours
- `"D"` - Daily
- `"W"` - Weekly

### Different Symbols

```python
# Ethereum
df = loader.fetch_history("ETHUSDT", "60", start_date, end_date)

# Solana
df = loader.fetch_history("SOLUSDT", "60", start_date, end_date)

# Any USDT perpetual futures pair
df = loader.fetch_history("ADAUSDT", "60", start_date, end_date)
```

## 🎨 Features

### Data Ingestion
- ✅ Automatic pagination (handles >1000 candles)
- ✅ Rate limit compliance (100ms delays)
- ✅ Clean data conversion to pandas DataFrame
- ✅ Efficient parquet storage
- ✅ Robust error handling

### Visualization
- 📈 Interactive candlestick charts
- 📊 Volume bar charts (color-coded)
- 📉 Technical indicators (20-period SMA)
- 🌙 Professional dark theme
- 🖱️ Zoom and pan controls
- 💡 Unified hover tooltips
- 🌐 Auto-opens in web browser

## ⚠️ Troubleshooting

### API Error 403

If you see this error:

```
API ACCESS WARNING
Bybit API returned an access error (403).
```

**Solutions**:
1. Use a VPN connected to a non-US location
2. Wait a few minutes if rate-limited
3. Check your internet connection

### No Data Returned

- Verify the symbol exists on Bybit (e.g., "BTCUSDT")
- Check date ranges are valid
- Ensure you're using `category="linear"` for USDT perpetuals

## 📁 File Structure

```
quanta futures/
├── bybit_futures_fetcher.py    # Main script
├── demo_visualization.py        # Demo with synthetic data
├── btc_futures.parquet          # Cached data (after first run)
└── chart_BTCUSDT.html          # Generated interactive chart
```

## 🔧 Customization

### Change SMA Period

Edit `visualize_data()` function:

```python
# 50-period SMA instead of 20
df['SMA_50'] = df['close'].rolling(window=50).mean()
```

### Add Multiple Indicators

```python
df['SMA_20'] = df['close'].rolling(window=20).mean()
df['SMA_50'] = df['close'].rolling(window=50).mean()
df['EMA_12'] = df['close'].ewm(span=12).mean()

# Add to chart
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_20'], name='SMA 20'))
fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_50'], name='SMA 50'))
```

### Export to CSV

```python
df.to_csv('btc_futures.csv', index=False)
```

## 📝 Data Format

The DataFrame contains:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Candle timestamp |
| `open` | float | Opening price |
| `high` | float | Highest price |
| `low` | float | Lowest price |
| `close` | float | Closing price |
| `volume` | float | Trading volume |

## 🚦 Rate Limits

The script includes:
- 100ms delay between requests
- Maximum 1000 candles per request (automatic pagination)
- Graceful error handling for rate limit violations

## 🌐 API Documentation

Official Bybit API docs: https://bybit-exchange.github.io/docs/v5/market/kline

## 📄 License

This project is provided as-is for educational and research purposes.

---

**Created by**: Senior Quantitative Developer  
**Version**: 1.0.0  
**Last Updated**: November 2025
=======
# QCF
>>>>>>> 0bcb7c5e1344c600665379be628567e16fd0c018
