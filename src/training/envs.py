import gym
import pandas as pd
from gym_anytrading.envs import TradingEnv

class GMETradingEnv(TradingEnv):
    def __init__(self, df, window_size, frame_bound):
        # Initialize the parent class with the provided DataFrame, window size, and frame bounds.
        super().__init__(df, window_size=window_size, frame_bound=frame_bound)

        # Process the data and initialize attributes used in the class.
        self._process_data()
        self.data = self.prices 

    def _process_data(self):
        # In this method, you can add any preprocessing of the data you need.
        # For example, you could create technical indicators or normalize the data.
        # For simplicity, this example only extracts the 'Close' prices and 'Volume'.
        self.prices = self.df[['Open', 'High', 'Low', 'Close']].values
        self.signal_features = self.df[['Volume']].values
        self._update_price_history()  # If the base TradingEnv uses price history for rendering etc.

    def _update_price_history(self):
        # This method can be used to update any internal state that relies on the price history.
        # It's not part of the gym_anytrading API but it's a placeholder for any additional setup you might need.
        pass

    
    def _calculate_reward(self, action):
    # Ensure current tick is valid to avoid IndexError
        if self._current_tick == 0 or self._current_tick >= self._data.shape[0]:
            return 0  # No previous price at the first tick or beyond dataset

        current_price = self._data[self._current_tick, 3]  # Access current close price using 'Close' column index
        previous_price = self._data[self._current_tick - 1, 3]  # Access previous close price

        # Define your reward function logic here
        reward = 0
        # Example logic for holding, buying, or selling
        if action == 0:  # Hold
            reward = 0
        elif action == 1:  # Buy
            reward = max(0, current_price - previous_price)  # Reward is the increase in price
        elif action == 2:  # Sell
            reward = max(0, previous_price - current_price)  # Reward is the decrease in price

        return reward

    def _get_observation(self, window_size):
        # This method should return the observation for the current tick.
        # It could be as simple as returning the last 'window_size' close prices.
        start = max(self.current_tick - window_size, 0)
        end = self.current_tick + 1
        return self.signal_features[start:end]

    def _done(self):
        # This method should return True if the episode is over, otherwise False.
        # In this case, it's when the current tick reaches the end of the data frame.
        return self.current_tick >= len(self.prices) - 1

    def _reset(self):
        # This method should reset any internal state and start a new episode.
        # It's a placeholder and should be tailored to your specific needs.
        pass

    def render(self, mode='human'):
        # This method can be used to render the environment.
        # The gym_anytrading environments have a built-in method for this,
        # but you can customize it here if needed.
        super().render(mode=mode)
