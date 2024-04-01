
import torch as nn
import torch.nn.functional as F
import torch.optim as optim

class VanillaDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        """
        Initializes the VanillaDQN class with the given state and action sizes.
        
        Args:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
        """
        super(VanillaDQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.model = nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size),
            nn.Softmax(dim=1)
        )

        self.loss = nn.MSELoss()

        self.optimizer= optim.Adam(self.model.parameters(), lr=0.001)
        self.epochs = 10
        self.gamma = 0.99

    def forward(self, x):
        """
        Method to forward input x through the model and return the result.
        
        Args:
            self: The instance of the class.
            x: The input data to be forwarded through the model.
            
        Returns:
            The result of forwarding the input data through the model.
        """
        return self.model(x)
 
    