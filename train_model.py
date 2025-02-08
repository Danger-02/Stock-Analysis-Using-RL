import yfinance as yf
import numpy as np
import pandas as pd
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

# Step 1: Fetch Nifty 50 data using yfinance
def fetch_nifty50_data():
    nifty50_tickers = [
        'RELIANCE.NS', 'TCS.NS', 'HDFCBANK.NS', 'INFY.NS', 'HINDUNILVR.NS',
        'ICICIBANK.NS', 'KOTAKBANK.NS', 'AXISBANK.NS', 'LT.NS', 'SBIN.NS'
    ]
    data = yf.download(nifty50_tickers, start="2020-01-01", end="2025-02-07", group_by='ticker')
    return data

# Step 2: Fetch fundamental data for a given ticker
def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    fundamental_data = {
        'PE_Ratio': info.get('trailingPE', np.nan),  # P/E Ratio
        'EPS': info.get('trailingEps', np.nan),  # Earnings Per Share
        'Dividend_Yield': info.get('dividendYield', np.nan),  # Dividend Yield
        'Revenue_Growth': info.get('revenueGrowth', np.nan),  # Revenue Growth
        # 'Debt_to_Equity': info.get('debtToEquity', np.nan),  # Debt to Equity Ratio
    }
    return fundamental_data

# Step 3: Preprocess data and calculate technical indicators
def preprocess_data(data):
    processed_data = {}
    for ticker in data.columns.levels[0]:
        df = data[ticker].copy()
        # Calculate technical indicators
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        df['RSI'] = calculate_rsi(df['Close'], 14)
        df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
        df = df.dropna()

        # Fetch fundamental data
        fundamental_data = fetch_fundamental_data(ticker)
        for key, value in fundamental_data.items():
            df[key] = value  # Add fundamental data to the DataFrame

        processed_data[ticker] = df
    return processed_data

def calculate_rsi(series, period):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, short_window=12, long_window=26, signal_window=9):
    short_ema = series.ewm(span=short_window, adjust=False).mean()
    long_ema = series.ewm(span=long_window, adjust=False).mean()
    macd = short_ema - long_ema
    signal = macd.ewm(span=signal_window, adjust=False).mean()
    return macd, signal

# Step 4: Create a custom Gym environment for stock trading
class StockTradingEnv(gym.Env):
    def __init__(self, df):
        super(StockTradingEnv, self).__init__()
        self.df = df
        self.n_step = len(df)
        self.current_step = 0

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # State space: Technical + Fundamental indicators
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(df.columns),), dtype=np.float32
        )

        # Initialize state
        self.state = self.df.iloc[self.current_step].values

    def reset(self):
        self.current_step = 0
        self.state = self.df.iloc[self.current_step].values
        return self.state

    def step(self, action):
        self.current_step += 1
        if self.current_step >= self.n_step - 1:
            done = True
            next_state = self.state
        else:
            done = False
            next_state = self.df.iloc[self.current_step].values

        current_price = self.df.iloc[self.current_step]['Close']
        if self.current_step < self.n_step - 1:
            next_price = self.df.iloc[self.current_step + 1]['Close']
        else:
            next_price = current_price

        reward = 0
        if action == 1:  # Buy
            reward = (next_price - current_price) / current_price
        elif action == 2:  # Sell
            reward = (current_price - next_price) / current_price

        self.state = next_state
        return self.state, reward, done, {}

# Step 5: Train the PPO model and save it for each stock
def train_and_save_model():
    data = fetch_nifty50_data()
    processed_data = preprocess_data(data)
    
    # Loop over each stock in the Nifty 50 list
    for ticker, df in processed_data.items():
        print(f"Training model for {ticker}...")
        env = DummyVecEnv([lambda: StockTradingEnv(df)])
        model = PPO('MlpPolicy', env, verbose=1, learning_rate=0.0003, n_steps=2048, batch_size=64, n_epochs=10, device='cuda')
        model.learn(total_timesteps=1000)
        model.save(f"ppo_stock_trading_model_{ticker}")
        print(f"Model for {ticker} trained and saved successfully.")

if __name__ == "__main__":
    train_and_save_model()