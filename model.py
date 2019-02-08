import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.hidden_1 = nn.Linear(state_size, 64)
        self.hidden_2 = nn.Linear(64,64)
        self.output = nn.Linear(64,action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.hidden_1(state))
        x = F.relu(self.hidden_2(x))
        
        return self.output(x)

class DuelQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size = action_size
        
        self.hidden_A_1 = nn.Linear(state_size, 32)
        self.hidden_A_2 = nn.Linear(32,32)
        self.output_A = nn.Linear(32,action_size)

        self.hidden_V_1 = nn.Linear(state_size, 32)
        self.hidden_V_2 = nn.Linear(32, 32)
        self.output_V = nn.Linear(32,1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        A = F.relu(self.hidden_A_1(state))
        A = F.relu(self.hidden_A_2(A))
        A = self.output_A(A)

        V = F.relu(self.hidden_V_1(state))
        V = F.relu(self.hidden_V_2(V))
        V = self.output_V(V).expand(state.size(0), self.action_size)
        
        return V + A - A.mean(1).unsqueeze(1).expand(state.size(0), self.action_size)