# Paper: WAVENET: A GENERATIVE MODEL FOR RAW AUDIO
from torch import nn
import torch 
import numpy as np
    

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.filter_conv = nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, bias=False)
        self.gated_conv = nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, bias=False)
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1, dilation=1, bias=False)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1, dilation=1, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        tanh_out = self.tanh(self.filter_conv(x))
        sig_out = self.sigmoid(self.gated_conv(x))
        comb_out = tanh_out * sig_out
        skip_out = self.skip_conv(comb_out)
        res_out = self.res_conv(comb_out) 
        res_out = res_out + x[..., -res_out.shape[-1]:] # residual connection, x is longer than res_out
        return res_out, skip_out


class StackResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, stack_size, layer_size):
        super(StackResidualBlock, self).__init__()
        dilations = np.array([2 ** i for i in range(layer_size)] * stack_size) # repeat the dilation values by stack_size times
        self.res_blocks = []
        for dilation in dilations:
            self.res_blocks.append(ResidualBlock(res_channels, skip_channels, kernel_size, dilation))

    def forward(self, x):
        res_out = x
        skip_outputs = []
        for res_b in self.res_blocks:
            res_out, skip_out = res_b(res_out)
            skip_outputs.append(skip_out)

        return res_out, sum([s[...,-res_out.size(2):] for s in skip_outputs])
    

class WaveNet(nn.Module):
    def __init__(self, in_channels, res_channels, skip_channels, out_channels, kernel_size, stack_size, layer_size):
        super(WaveNet, self).__init__()
        
        self.in_conv = nn.Conv1d(in_channels, res_channels, kernel_size = 1, dilation=1, bias=False)
        self.stack_res_block = StackResidualBlock(res_channels, skip_channels, kernel_size, stack_size, layer_size)
        self.dense_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, out_channels, kernel_size=1, dilation=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=1, dilation=1, bias=False)
        )

        receptive_field = 1
        for b in range(stack_size):
            for i in range(layer_size):
                receptive_field += 2 ** i
        self.receptive_field = receptive_field

    def forward(self, x):
        x = self.in_conv(x)
        _, skip_connections = self.stack_res_block(x)
        return self.dense_layer(skip_connections)
             
