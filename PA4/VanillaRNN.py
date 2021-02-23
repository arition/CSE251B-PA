import torch
import torch.nn as nn
from torchvision import datasets, transforms, models

class VanillaRNN(nn.Module):

    def __init__(self, embedding_size, hidden_size, output_size):
        super(VanillaRNN, self).__init__()
        
        self.pre_trained_encoder = models.resnet50(pretrained=True)
        
        num_ftrs = self.pre_trained_encoder.fc.in_features
        
        # The output size of encoder is equal to input size of Vanilla RNN
        self.pre_trained_encoder.fc = nn.Linear(num_ftrs, embedding_size)
        
        # Freeze the parameters
        for parameter in self.pre_trained_encoder.parameters():
            parameter.requires_grad = False
            
        # Only train the last layer
        for parameter in self.pre_trained_encoder[-1].parameters():
            parameter.requires_grad = True
            
        # Decoding part
        self.I_to_H = nn.Linear(embedding_size + hidden_size, hidden_size)
        
        self.H_to_O = nn.Linear(hidden_size, output_size)
        
        # Init hidden state
        self.hidden = torch.zeros(1, self.hidden_size)
        
        
    def forward(self, x):
        x = self.pre_trained_encoder(x)
        
        x = torch.cat((x, self.hidden), 1)
        
        # Input+Hidden to Hidden
        x = self.I_to_H(x)
        
        # Activation
        self.hidden = nn.Tanh(x)
        
        # Hidden to Output
        x = self.H_to_O(self.hidden)
        
        return x

