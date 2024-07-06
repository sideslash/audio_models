
import torch
import torch.nn as nn
import wavenet
import numpy as np

class SubModel(nn.Module):
    def __init__(self):
        super(SubModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def forward(self, x):
        print(f"x type:{type(x)}")
        print(f"weight type:{type(self.conv1.weight)}")
        return self.conv1(x)

class WrapperModel(nn.Module):
    def __init__(self, count):
        super(WrapperModel, self).__init__()
        self.blocks = []
        for i in range(count):
            self.blocks.append(SubModel())

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
     
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2)
        self.wrapper = WrapperModel(2)
        

    def forward(self, x):
        x = self.conv(x)
        x = self.wrapper(x)
        
        return x


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)
# model = MyModel()
# model.to(device)
# # model.wrapper.to(device)
# input_tensor = torch.randn(1, 1, 10).to(device)  
# output = model(input_tensor)

np.random.seed(0)
x = np.random.randint(0, 8, (1, 10))
print(x)
x = torch.from_numpy(x)
print(x)
one_hot = torch.zeros(8, x.shape[1], dtype=torch.float32)
print(one_hot)
one_hot.scatter_(0, x.to(torch.int64), 1)
print(one_hot)

