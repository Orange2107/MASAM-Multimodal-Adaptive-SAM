
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)

def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)

def fix_transformer_dropout(module):
    dropout_masks = {}

    def apply_fixed_dropout_mask(dropout_module, input):
        if dropout_module not in dropout_masks:
            dropout_masks[dropout_module] = (torch.rand_like(input) > dropout_module.p).float().detach() / (1 - dropout_module.p)
        return input * dropout_masks[dropout_module]

    for layer in module.modules():
        if isinstance(layer, nn.TransformerEncoderLayer):
            layer.dropout.forward = lambda input, mod=layer.dropout: apply_fixed_dropout_mask(mod, input)
            layer.dropout1.forward = lambda input, mod=layer.dropout1: apply_fixed_dropout_mask(mod, input)
            layer.dropout2.forward = lambda input, mod=layer.dropout2: apply_fixed_dropout_mask(mod, input)

def freeze_layernorm(model):
    for module in model.modules():
        if isinstance(module, nn.LayerNorm):
            module.eval()

def restore_transformer_dropout(model):
    for module in model.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            for sub_module in module.children():
                if isinstance(sub_module, nn.Dropout):
                    sub_module.forward = nn.Dropout(sub_module.p).forward

