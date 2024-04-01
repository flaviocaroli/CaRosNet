import random
from collections import deque


class ReplayBuffer:
    def __init__(self, buffer_size):
        """
        Initializes the object with the specified buffer size.

        Args:
            buffer_size (int): The size of the buffer.

        Returns:
            None
        """    
        self.buffer = deque(maxlen=buffer_size)

    def push(self, state, action, reward, next_state, done):
        """
        Add the given state, action, reward, next_state, and done tuple to the buffer.
        
        Args:
            state (object): the current state
            action (object): the action taken
            reward (float): the reward received
            next_state (object): the next state
            done (bool): whether the episode is done
        
        Returns:
            None
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Selects a random sample of batch_size elements from the buffer and returns it.

        Args:
            batc_size(int): size of the batch
        Returns:
            random sample of batch_size elements from the buffer
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Return the length of the buffer.
        """
        return len(self.buffer)
    
#################################
    #HYPERPARAMETER TUNING#
#################################
    

    #OPTUNA
    #GRIDSEARCH
    #RANDOMSEARCH