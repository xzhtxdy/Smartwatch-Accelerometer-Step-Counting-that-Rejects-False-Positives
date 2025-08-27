import torch
import torch.nn as nn
import torch.nn.functional as F


class NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=32, num_layers=1, dropout=0.0):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_layers - 1):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for i in range(self.num_layers):
            x = F.relu(self.layers[i](x))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.layers[-1](x)
        return x


class CNN(nn.Module):
    def __init__(self, input_dim=3, output_dim=4, hidden_dim=32, dropout=0.5):
        super(CNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.hidden_dim,
                      kernel_size=6, stride=2),
            nn.BatchNorm1d(self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.cnn2 = nn.Sequential(
            nn.Conv1d(self.hidden_dim, 4 * self.hidden_dim,
                      kernel_size=5, stride=2),
            nn.BatchNorm1d(4 * self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.cnn3 = nn.Sequential(
            nn.Conv1d(4 * self.hidden_dim, 6*self.hidden_dim,
                      kernel_size=5, stride=1),
            nn.BatchNorm1d(6*self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout))
        self.linear = nn.Sequential(
            nn.Linear(self.hidden_dim*6, self.hidden_dim),
            nn.ReLU(),
        )
        self.linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.cnn(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = x.squeeze()
        x = self.linear(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x.squeeze()
