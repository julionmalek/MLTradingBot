# Advanced Machine Learning Trading Algorithm

## Overview
This trading algorithm utilizes **Lumibot**, **Alpaca API**, and **technical indicators** to execute trades based on sentiment analysis, momentum indicators, and risk allocation strategies. The bot dynamically adjusts risk exposure based on **SPY RSI values** and invests in both **stable stocks** (long-term growth) and **volatile stocks** (high-risk, high-reward potential).

## How It Works
### 1. **Initialization**
- The bot initializes with a set **list of stocks**.
- It allocates cash according to a **proportions dictionary**.
- It ensures that **30% of the cash is initially invested in SPY** to maintain stability.
- The bot sets a **risk tolerance level** based on the market conditions.

### 2. **Position Sizing**
- Determines how much cash should be allocated per stock.
- Uses the **last price of the stock** to calculate the number of shares to purchase.

### 3. **Risk Management (Dynamic Allocation)**
- Uses **Relative Strength Index (RSI) of SPY** to **adjust the risk**:
  - **RSI > 70** → Risk is maintained at **high levels**.
  - **RSI < 30** → Risk is increased for potential upside.
  - **RSI between 30-70** → Default **high-risk** approach.
- This ensures the strategy adapts to market conditions dynamically.

### 4. **Sentiment Analysis (News-Based Trading)**
- Retrieves financial news for each stock.
- Uses **FinBERT (a financial sentiment analysis model)** to determine:
  - **Probability of positive sentiment**
  - **Classification: positive, neutral, or negative**
- If sentiment is highly positive, the bot considers **buying**.
- If sentiment is strongly negative, the bot considers **selling**.

### 5. **Technical and Momentum Indicators**
#### **Technical Indicators**
- **Relative Strength Index (RSI)**
  - Measures stock momentum (overbought vs. oversold conditions).
  - **RSI > 70** → Stock is overbought (possible sell signal).
  - **RSI < 30** → Stock is oversold (possible buy signal).
- **Simple Moving Averages (SMA 20 & 50)**
  - **SMA 20 > SMA 50** → Uptrend (buy signal).
  - **SMA 20 < SMA 50** → Downtrend (sell signal).

#### **Momentum Indicators**
- **Moving Average Convergence Divergence (MACD)**
  - **MACD > MACD Signal** → Bullish momentum (buy signal).
  - **MACD < MACD Signal** → Bearish momentum (sell signal).
- **Average Directional Index (ADX)**
  - Measures the **strength** of a trend (higher values mean a stronger trend).
  - **ADX > 25** → Strong trend confirmation.
  - **ADX < 20** → Weak trend, avoid trading.

### 6. **Execution of Trades**
#### **Buy Criteria**
- Positive sentiment with high probability (> 80%)
- RSI below 70 (not overbought)
- SMA 20 > SMA 50 (uptrend confirmation)
- MACD > MACD Signal (momentum confirmation)
- ADX > 25 (strong trend confirmation)

#### **Sell Criteria**
- Negative sentiment with high probability (> 70%)
- RSI above 85 (overbought, likely to drop)
- SMA 20 < SMA 50 (downtrend confirmation)
- MACD < MACD Signal (momentum confirmation)
- ADX > 20 (downtrend confirmation)

### 7. **Trailing Stop Loss & Profit Taking**
- Uses **trailing stop orders** to **lock in profits while letting winners run**.
- **5% trailing stop for all trades** to protect capital.
- **30% trailing stop for SPY** to maintain stability.

## **Selected Stocks**
### **Stable Performers** (Low-Risk, Strong Growth)
1. **AAPL (Apple)**
2. **MSFT (Microsoft)**
3. **GOOGL (Google)**
4. **AMZN (Amazon)**
5. **SPY (S&P 500 ETF)**

### **Volatile Stocks** (High-Risk, High-Reward)
6. **TSLA (Tesla)**
7. **NVDA (NVIDIA)**
8. **PLTR (Palantir)**
9. **ARKK (ARK Innovation ETF)**
10. **SQ (Block, formerly Square)**

## Risk Strategy
- **Aggressive risk allocation** (up to 100% cash at risk).
- Higher weight given to **volatile stocks**.
- **Market-adaptive risk levels** based on SPY’s RSI.

## **Backtesting Configuration**
- **Backtesting Library**: Lumibot
- **Data Source**: Yahoo Finance (YahooDataBacktesting)
- **Broker API**: Alpaca
- **Timeframe**: Jan 2023 - Dec 2024
- **Sleep time between trades**: 12 hours (increases trading frequency)

## Running the Algorithm
### **Installation**
Ensure you have the required libraries installed:
```sh
pip install lumibot alpaca-trade-api ta-lib numpy transformers
```

### **Executing the Script**
```sh
python tradingbotUpgraded.py
```

## Conclusion
This **advanced machine learning trading bot** combines **technical indicators, momentum signals, and sentiment analysis** to execute risk-adjusted trades. It is designed for **high-risk, high-reward strategies** while maintaining **SPY as a stable core investment**. The system dynamically **adapts risk based on SPY RSI** and aims to capture **high-growth opportunities** in volatile stocks.

