import yfinance as yf
import numpy as np
import gym
from stable_baselines3 import PPO
from gym import spaces
import os

# Step 1: Fetch historical stock data for the given ticker (company name)
def fetch_data_for_prediction(ticker):
    data = yf.download(ticker, start="2020-01-01", end="2023-10-01")
    return data

# Step 2: Preprocess data and calculate technical indicators
def preprocess_data_for_prediction(data):
    df = data.copy()
    # Calculate technical indicators
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['RSI'] = calculate_rsi(df['Close'], 14)
    df['MACD'], df['MACD_signal'] = calculate_macd(df['Close'])
    df = df.dropna()
    
    # Add a dummy fundamental data for simplicity (you can add actual data here if needed)
    df['PE_Ratio'] = 20  # Dummy value
    df['EPS'] = 5  # Dummy value
    df['Dividend_Yield'] = 1  # Dummy value
    df['Revenue_Growth'] = 0.1  # Dummy value
    return df

# Reuse RSI and MACD calculation functions
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

# Step 3: Create a custom Gym environment for stock trading
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

# Step 4: Load the trained model and make predictions
def make_prediction(ticker):
    # Load the trained PPO model for the stock
    model_path = f"ppo_stock_trading_model_{ticker}"
    
    if not os.path.exists(model_path):
        print(f"Model for {ticker} not found.")
        return
    
    model = PPO.load(model_path)
    
    # Fetch and preprocess the data for prediction
    data = fetch_data_for_prediction(ticker)
    processed_data = preprocess_data_for_prediction(data)
    
    # Create environment
    env = StockTradingEnv(processed_data)
    
    # Reset the environment and get the initial state
    state = env.reset()
    
    # Get model's action prediction
    action, _states = model.predict(state)
    
    # Determine the action based on model output
    if action == 1:
        print(f"Action for {ticker}: BUY")
    elif action == 2:
        print(f"Action for {ticker}: SELL")
    else:
        print(f"Action for {ticker}: HOLD")

# Step 5: Main function to take user input and predict
if __name__ == "__main__":
    company_name = input("Enter company name (ticker, e.g., 'RELIANCE.NS'): ").strip()
    make_prediction(company_name)