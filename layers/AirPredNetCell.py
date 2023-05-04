import torch
import torch.nn as nn
from torch.nn import init
class AirPredNetCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super().__init__()
        padding = filter_size // 2
        self.reset_gate = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden, filter_size, padding=padding),
            nn.LayerNorm([num_hidden, width, width]))
        self.update_gate = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden, filter_size, padding=padding),
            nn.LayerNorm([num_hidden, width, width]))
        self.out_gate = nn.Sequential(
            nn.Conv2d(in_channel + num_hidden, num_hidden, filter_size, padding=padding),
            nn.LayerNorm([num_hidden, width, width]))
        
        init.orthogonal_(self.reset_gate[0].weight)
        init.orthogonal_(self.update_gate[0].weight)
        init.orthogonal_(self.out_gate[0].weight)
        init.constant_(self.reset_gate[0].bias, 0.)
        init.constant_(self.update_gate[0].bias, 0.)
        init.constant_(self.out_gate[0].bias, 0.)
        
    def forward(self, input_, prev_state):
        # data size is [batch, channel, height, width]
        stacked_inputs = torch.cat([input_, prev_state], dim=1)
        update = torch.sigmoid(self.update_gate(stacked_inputs))
        reset = torch.sigmoid(self.reset_gate(stacked_inputs))
        out_inputs = torch.tanh(self.out_gate(torch.cat([input_, prev_state * reset], dim=1)))
        new_state = prev_state * (1 - update) + out_inputs * update
        return new_state