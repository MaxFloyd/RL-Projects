import torch.nn as nn
import torch.nn.functional as F

class DQ_Network(nn.Module):
    def __init__(self, state_size, action_size, fc1_lay, fc2_lay, fc3_lay):
        """
        state_size (int) - dim of state vector (input)
        action_size (int) - dim of action vector (output)
        fc1_lay, fc2_lay, fc3_lay - number of units in each linear layer 
        """
        super().__init__()
        
        self.fc1 = nn.Linear(state_size, fc1_lay)
        self.fc2 = nn.Linear(fc1_lay, fc2_lay)
        self.fc3 = nn.Linear(fc2_lay, fc3_lay)
        self.out = nn.Linear(fc3_lay, action_size)
        
    def forward(self, inpt):
        
        x = F.relu(self.fc1(inpt))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        return self.out(x)
        
        