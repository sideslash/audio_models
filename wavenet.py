# Paper: WAVENET: A GENERATIVE MODEL FOR RAW AUDIO
from torch import nn
import torch 
import numpy as np


# class CausalDilatedConv1D(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=2, padding = 0, dilation=1):
#         super().__init__()
#         pad = (kernel_size - 1) * dilation # use the padding to make the conv causal
#         # print(f"CausalDilatedConv1D_init in_channels:{in_channels} out_channels:{out_channels} kernel_size:{kernel_size} dilation:{dilation} pad:{pad}")
#         self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation, bias=False)
#         # self.conv1D.weight.data = torch.tensor([[[1.0, 1.0, 1.0]]])

#     def forward(self, x):
#         # # print(f"CausalDilatedConv1D_forward x:{x.shape}")
#         # out = self.conv1D(x)
#         # # out = out[..., :-self.conv1D.padding[0]]
#         # # out = out[..., :-self.ignore]
#         # # print(f"CausalDilatedConv1D_forward x:{x.shape} out:{out.shape} self.conv1D.padding[0]:{self.conv1D.padding[0]}")
#         # if self.conv1D.padding[0] == 0:
#         #     return out
#         # else:
#         #     return out[..., :-self.conv1D.padding[0]]
#         # # print(f"out2:{out.shape}")

    

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        # print(f"ResidualBlock_init res_channels:{res_channels} skip_channels:{skip_channels} kernel_size:{kernel_size} dilation:{dilation}")
        self.filter_conv = nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, bias=False)
        self.gated_conv = nn.Conv1d(res_channels, res_channels, kernel_size, dilation=dilation, bias=False)
        self.res_conv = nn.Conv1d(res_channels, res_channels, kernel_size=1, dilation=1, bias=False)
        self.skip_conv = nn.Conv1d(res_channels, skip_channels, kernel_size=1, dilation=1, bias=False)
        # print(f"ResBlock_init res_channels:{res_channels} skip_channels:{skip_channels} kernel_size:{kernel_size} dilation:{dilation}")
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # print(next(self.filter_conv.parameters()).device)
        # print(next(self.gated_conv.parameters()).device)
        # print(next(self.res_conv.parameters()).device)
        # print(next(self.skip_conv.parameters()).device)
        # print(f"ResBlock_forward: x_shape:{x.shape}")
        # print(f"ResidualBlock_forward x:{x.shape} kernel_size:{self.filter_conv.kernel_size}")
        tanh_out = self.tanh(self.filter_conv(x))
        sig_out = self.sigmoid(self.gated_conv(x))
        comb_out = tanh_out * sig_out
        skip_out = self.skip_conv(comb_out)
        res_out = self.res_conv(comb_out) 
        res_out = res_out + x[..., -res_out.shape[-1]:] # residual connection, x is longer than res_out
        return res_out, skip_out


class StackResidualBlock(nn.Module):
    def __init__(self, res_channels, skip_channels, kernel_size, stack_size, layer_size):
        super().__init__()
        # vectorized_dilation_func = np.vectorize(self.buildDilations)
        # dilations = vectorized_dilation_func(stack_size, layer_size)
        dilations = np.array([2 ** i for i in range(layer_size)] * stack_size) # repeat the dilation values by stack_size times
        self.res_blocks = []
        for dilation in dilations:
            self.res_blocks.append(ResidualBlock(res_channels, skip_channels, kernel_size, dilation))

    def forward(self, x):
        # print(f"StackResBlock forward: x.shape:{x.shape}")
        res_out = x
        skip_outputs = []
        for res_b in self.res_blocks:
            res_out, skip_out = res_b(res_out)
            skip_outputs.append(skip_out)
            # print(f"skip_out.shape:{skip_out.shape}")

        # return res_out, torch.stack(skip_outputs)
        return res_out, sum([s[...,-res_out.size(2):] for s in skip_outputs])
    

class WaveNet(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size, stack_size, layer_size):
        super().__init__()
        
        self.in_conv = nn.Conv1d(in_channels, in_channels, kernel_size = 1, dilation=1, bias=False)
        self.stack_res_block = StackResidualBlock(in_channels, skip_channels, kernel_size, stack_size, layer_size)
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
        # print(next(self.in_conv.parameters()).device)
        # print(next(self.stack_res_block.parameters()).device)
        # print(next(self.dense_layer.parameters()).device)
        x = self.in_conv(x)
        _, skip_connections = self.stack_res_block(x)
        return self.dense_layer(skip_connections)
             

# convs = []
# for i in range(3):
#     convs.append(nn.Conv1d(1, 1, 2, padding=0, dilation=2**i, bias=False))
#     convs[i].weight.data = torch.tensor([[[1.0, 1.0]]])

# x = torch.tensor([[[1,2,3,4,5,6,7,8]]], dtype=torch.float32)
# print(x)
# for i in range(3):
#     x = convs[i](x)
#     print(x)

# x = torch.tensor([[[2,3,4,5,6,7,8,9]]], dtype=torch.float32)
# print(x)
# for i in range(3):
#     x = convs[i](x)
#     print(x)

# x = torch.tensor([[[1,2,3,4,5,6,7,8,9]]], dtype=torch.float32)
# print(x)
# for i in range(3):
#     x = convs[i](x)
#     print(x)



# # torch stack function test
# tensor1 = torch.tensor([1, 2, 3])
# tensor2 = torch.tensor([4, 5, 6])
# tensor3 = torch.tensor([7, 8, 9])
# # Stack the tensors along a new dimension
# stacked_tensor = torch.stack([tensor1, tensor2, tensor3], dim=0)
# print(stacked_tensor)
# stacked_tensor = torch.stack([tensor1, tensor2, tensor3], dim=1)
# print(stacked_tensor)
# stacked_tensor = torch.stack([tensor1, tensor2, tensor3], dim=-1)
# print(stacked_tensor)
# stacked_tensor = torch.stack([tensor1, tensor2, tensor3], dim=-2)
# print(stacked_tensor)

# print([2 ** d for d in range(4)] * 3)
# print(np.array([1,2,3,4,5,6,7,8,9,10])[-2:])

# layers = 4
# stacks = 1
# dilations = [2**i for i in range(layers)] * stacks
# print(dilations)
# receptive_field = 1
# for b in range(stacks):
#     for i in range(layers):
#         receptive_field += 2 ** i
# print(receptive_field)

# print(stacks * (2 ** layers * 2) - (stacks - 1))

# pad = (2 - 1) * 1 
# conv1D = nn.Conv1d(1, 1, 2, padding=pad, dilation=1, bias=False)
# conv1D.weight.data = torch.tensor([[[1.0, 1.0]]])
# x = torch.tensor([[[1,2,3,4,5,6,7,8,9,10]]], dtype=torch.float32)
# print(conv1D(x)[..., :-pad])

# conv2 = nn.Conv1d(1, 1, 2, padding=0, dilation=1, bias=False)
# conv2.weight.data = torch.tensor([[[1.0, 1.0]]])
# print(conv2(x))