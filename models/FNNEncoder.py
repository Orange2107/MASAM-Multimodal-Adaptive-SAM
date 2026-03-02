import torch

import torch.nn as nn


class FFNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout_prob=0.5, device="cuda"):
        super(FFNEncoder, self).__init__()

        num_middle_layers = num_layers - 2
        print(f"num_layers: {num_layers}")
        assert num_middle_layers >= 0, f"Number of layers must be at least 2, got {num_layers}"

        self.dropout0 = nn.Dropout(p=dropout_prob)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout_prob)

        self.fcs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.relus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(num_middle_layers):
            self.fcs.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            self.relus.append(nn.ReLU())
            self.dropouts.append(nn.Dropout(p=dropout_prob))
        self.fc2 = nn.Linear(hidden_dim, output_dim)

        self.device = device

    def forward(self, x):
        x = x.to(self.device)

        x = self.dropout0(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        for fc, bn, relu, dropout in zip(self.fcs, self.bns, self.relus, self.dropouts):
            x = fc(x)
            x = bn(x)
            x = relu(x)
            x = dropout(x)
        x = self.fc2(x)
        return x