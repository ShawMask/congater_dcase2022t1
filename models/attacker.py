import torch.nn as nn


class AttackerModel(nn.Module):
    def __init__(self, embedding_size, hidden_layers, num_attributes, activation_function='ReLU'):
        super(AttackerModel, self).__init__()

        self.layers = nn.ModuleList()
        if not hidden_layers is None:
            layer_size = [embedding_size] + hidden_layers
            layer_range = len(layer_size) - 1
            for i in range(layer_range):
                fc = nn.Linear(layer_size[i], layer_size[i + 1])
                if activation_function == 'ReLU':
                    act_function = nn.ReLU()
                self.layers.append(nn.Sequential(fc, act_function))
        else:
            layer_size = [embedding_size]

        self.out = nn.Linear(layer_size[-1], num_attributes)


    def forward(self, x):
        for module in self.layers:
            x = module(x)
        out = self.out(x)
        return out
