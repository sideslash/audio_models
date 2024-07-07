import numpy as np
import torch
from torch import nn
import os
import logging
from dataset import WaveNetDataset
from wavenet import WaveNet
from hparams import HParams

class hparams(HParams):
    stack_size: int  = 10
    layer_Size: int  = 3

    in_out_channels: int  = 256
    res_channel: int  = 64
    skip_channel: int  = 512
    
    output_length: int = 128

    epochs: int = 30
    learning_rate: float = 0.001
    batch_size: int = 32

    decay_start: int = 10
    lr_decay: float = learning_rate / 10.0    
    lr_update_epoch: int = 1

    model_name: str = 'wavenet'


def adjust_learning_rate(optimizer, lr, decay):
    new_lr = max(0.000005, lr - decay)
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    return new_lr, optimizer

 
if __name__ == "__main__":
    file_log = "WaveNet.log"

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS backend is available.")
    elif torch.cuda.is_available():
        print(f"CUDA is available, devices count {print(torch.cuda.device_count())}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("MPS and CUDA is not available. Using CPU instead.")

    param = hparams()

    model = WaveNet(param.in_out_channels, 
                    param.res_channel, 
                    param.skip_channel, 
                    param.in_out_channels, 
                    kernel_size=2, 
                    stack_size=param.stack_size, 
                    layer_size=param.layer_Size)
    model = model.to(device)
    receptive_field = model.receptive_field
    print(f"receptive_field:{receptive_field}, output_length:{param.output_length}")

    opttimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate, betas=[0.5, 0.999])

    cross_loss = nn.CrossEntropyLoss()

    # Dataset
    dataset = WaveNetDataset(receptive_field, param.output_length)
    print(f"len(dataset):{len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size, shuffle=True, num_workers=8)

    for epoch in range(param.epochs):
        model.train()
        
        if epoch > param.decay_start and (epoch - param.decay_start) % param.lr_update_epoch == 0:
            param.learning_rate, opttimizer = adjust_learning_rate(opttimizer, param.learning_rate, param.lr_decay)
            logging.info(f"learning rate is updated to {param.learning_rate}")

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            opttimizer.zero_grad()
            y_pred = model(x)
            # print(f"step1: y_pred:{y_pred.shape}, y:{y.shape}, x:{x.shape}")

            y = y.view(-1)
            y_pred = y_pred.permute(0, 2, 1).contiguous()
            y_pred = y_pred.view(-1, y_pred.size(-1))

            loss = cross_loss(y_pred, y)
            # print(f"loss:{loss.item()}")

            loss.backward()
            opttimizer.step()

            if i % 100 == 0:
                print(f"epoch:{epoch}, iteration:{i}, loss:{loss.item()}")

        