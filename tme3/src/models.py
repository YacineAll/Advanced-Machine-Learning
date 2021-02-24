
import torch.nn as nn
from torch.autograd import Function


class AutoEncoder(nn.Module):
    def __init__(self, in_features, latent_size):
        super().__init__()
        self.encodeur = nn.Sequential(
            nn.Linear(in_features, latent_size),
            nn.ReLU()
        )

        self.decodeur = nn.Sequential(
            nn.Linear(latent_size, in_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encodeur(x)
        x = self.decodeur(x)
        return x


class HighwayLayer(nn.Module):
    def __init__(self, size_in):
        super().__init__()
        self.H = nn.Linear(size_in, size_in)
        self.H.bias.data.zero_()

        self.T = nn.Linear(size_in, size_in)
        self.T.bias.data.fill_(-1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        H = self.relu(self.T(x))
        T = self.sigmoid(self.H(x))
        return H*T + x*(1-T)


class HighwayNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, number_of_layer=9):
        super().__init__()
        self.my_input_layer = nn.Linear(input_size, hidden_size)
        self.highways = nn.ModuleList(
            [HighwayLayer(hidden_size) for _ in range(number_of_layer)])
        self.my_output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.my_input_layer(x)
        for highway in self.highways:
            x = highway(x)
        return self.my_output_layer(x)
