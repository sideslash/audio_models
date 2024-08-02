import numpy as np
import torch
from torch import nn
import os
import logging
from dataset import WaveNetDataset
from wavenet import WaveNet
from param import Model_Params

def adjust_learning_rate(optimizer, lr, decay):
    new_lr = max(0.000005, lr - decay)
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr, optimizer

 
if __name__ == "__main__":
    file_log = "WaveNet.log"

    # Set up basic configuration for logging
    logging.basicConfig(level=logging.INFO, 
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.FileHandler(file_log), 
                                  logging.StreamHandler()])
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("MPS backend is available.")
    elif torch.cuda.is_available():
        print(f"CUDA is available, devices count {print(torch.cuda.device_count())}")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("MPS and CUDA is not available. Using CPU instead.")

    device = torch.device("cpu")
    param = Model_Params()

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

    optimizer = torch.optim.Adam(model.parameters(), lr=param.learning_rate, betas=[0.5, 0.999])

    cross_loss = nn.CrossEntropyLoss()

    # Dataset
    dataset = WaveNetDataset(receptive_field, param.output_length)
    print(f"len(dataset):{len(dataset)}")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=param.batch_size, shuffle=True, num_workers=8)

    checkpoint_folder = "checkpoints"
    os.makedirs(checkpoint_folder, exist_ok=True)

    for epoch in range(param.epochs):
        model.train()
        
        if epoch > param.decay_start and (epoch - param.decay_start) % param.lr_update_epoch == 0:
            param.learning_rate, optimizer = adjust_learning_rate(optimizer, param.learning_rate, param.lr_decay)
            logging.info(f"learning rate is updated to {param.learning_rate}")

        for i, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_pred = model(x)

            y = y.view(-1)
            y_pred = y_pred.permute(0, 2, 1).contiguous()
            y_pred = y_pred.view(-1, y_pred.size(-1))

            loss = cross_loss(y_pred, y)

            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                logging.info(f"epoch:{epoch}, iteration:{i}, loss:{loss.item()}")

        # save model
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }
        path = os.path.join(checkpoint_folder, f"checkpoint_{epoch}.pth")
        torch.save(checkpoint, path)



        