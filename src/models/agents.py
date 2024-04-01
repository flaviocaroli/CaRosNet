
import numpy as np
import random
from utils import ReplayBuffer
from networks import VanillaCNN, VanillaDQN
import torch


class DQNAgent:
    def __init__(self, state_size, action_size, buffer_size, batch_size, gamma, epsilon, e_min, e_decay, lr):
        """
        Initializes the DQNAgent with the given parameters.

        Parameters:
            state_size (int): The size of the state space.
            action_size (int): The size of the action space.
            buffer_size (int): The size of the replay buffer.
            batch_size (int): The batch size for training.
            gamma (float): The discount factor for future rewards.
            epsilon (float): The exploration rate.
            e_min (float): The minimum exploration rate.
            e_decay (float): The decay rate for exploration.
            lr (float): The learning rate for the neural network.

        Returns:
            None
        """
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.e_min = e_min
        self.e_decay = e_decay
        self.memory = ReplayBuffer(self.buffer_size)
        self.policy_net = VanillaDQN(state_size = self.state_size , action_size= self.action_size)
        

    def act(self, state):
        """
        A function that takes in a state and returns an action based on epsilon-greedy policy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, self.actions - 1)
        else:
            with torch.no_grad():
                Q_values = self.policy_net(float(torch.from_numpy(state)))
                return Q_values.argmax.item()
    
    def learn(self):
    	
        if self.memory < self.batch_size:
            return
        
        #dones are boolean flag that indicates whether the current state is a terminal state or not.
        states, actions, rewards, next_states, dones = zip(*random.sample(self.memory.buffer, self.batch_size))
        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        #gather does: is used to gather the Q-values corresponding to the actions taken from the output of the policy network (self.policy_net(states)).
        Q_values = self.policy_net(states).gather(1, actions)

        #detach does: tensor operation that creates a new tensor that is detached from the current computational graph.
        target_q_values = self.target_net(next_states).max(dim=1)[0].detach()
        target_q_values[dones == 1] = 0

        #value iteration
        expected_q_values = rewards + self.gamma * target_q_values

        #squeeze tensor operation that removes all single-dimensional entries from the shape of a tensor.
        loss = self.policy_net.loss(Q_values.squeeze(), expected_q_values)
        opt = self.policy_net.optimizer
        opt.zero_grad()
        loss.backward()
        opt.step()

        if self.epsilon > self.e_min:
            self.epsilon = self.epsilon * self.e_decay

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

