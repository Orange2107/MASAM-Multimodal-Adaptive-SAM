
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import swin_s, Swin_S_Weights
from torchvision.models import swin_t, Swin_T_Weights
from sklearn.metrics import average_precision_score, roc_auc_score
import math
import os
import time




class _URTransformer(nn.Module):

    def __init__(self, n_features: int, dim: int, n_head: int, n_layers: int):
        super().__init__()
        self.embed_dim = dim
        self.conv = nn.Conv1d(n_features, self.embed_dim, kernel_size=1, padding=0, bias=False)
        layer = nn.TransformerEncoderLayer(self.embed_dim, nhead=n_head)
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x.permute(0, 2, 1))
        x = x.permute(2, 0, 1)
        x = self.transformer(x)[0]
        return x
