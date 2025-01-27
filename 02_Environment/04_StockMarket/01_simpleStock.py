import gymnasium as gym

from typing import Dict
import numpy as np
import os

env_config = {
    "ticker": "TSLA",
    "opening_balance": 1000,
    "observation_length": 30,
    "order_size": 1,
}

print(env_config.get("ticker", ))

class StockTradingEnv(gym.Env):
    def __init__(self, env_config: Dict = env_config):
        super(StockTradingEnv, self).__init__()

        self.ticker = env_config.get("ticker", "GOOGL")
        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
        self.ticker_file_stream = os.path.join(f"{data_dir}", f"{self.ticker}.csv")
        
        # data stream. offline file stream used. Alternatively, use online web API to pull live data.
        self.ohlcv_df = pd.read_csv(self.ticker_file_stream)    # Date, Open, High, Low, Close, Adj-Close, Volume

        self.opening_account_balance = env_config["opening_balance"]
       
        self.action_space = gym.spaces.Discrete(3)   # Hold, Buy, Sell;
        self.observation_features = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
        self.horizon = env_config.get("observation_length")
        
        self.observation_space = gym.spaces.Box(low=0, high=1,
            shape=(len(self.observation_features), self.horizon + 1), dtype=np.float32)
        self.order_size = env_config.get("order_size")
        self.viz = None  # Visualizer


    def get_observation(self):
        observation = (
            self.ohlcv_df.loc[ 
                self.current_step : self.current_step + self.horizon, 
                self.observation_features,].to_numpy().T
            )
        return observation