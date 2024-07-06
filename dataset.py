import torch
import numpy as np
import os
# import librosa
import matplotlib.pyplot as plt
import pandas as pd

class WaveNetDataset(torch.utils.data.Dataset):
    def __init__(self, receptive_field, output_length):
        super().__init__()
        self.receptive_field = receptive_field
        self.output_length = output_length

        # load data
        samples = []
        data_processed_path = "data_processed"
        audio_files = os.listdir(data_processed_path)
        print(f"audio_files:{audio_files}")
        for audio_file in audio_files:
            data = np.load(f"{data_processed_path}/{audio_file}")
            samples.append(data)
        self.samples = np.array(samples) 

    def __len__(self):
        return (self.samples.shape[1] - self.receptive_field) // self.output_length

    def __getitem__(self, index):
        subset_start = self.output_length * index
        subset_end = self.output_length * index + self.receptive_field + self.output_length

        x = self.samples[:, subset_start : subset_end-1]
        y = self.samples[:, subset_end - self.output_length : subset_end]
        
        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        one_hot_x = torch.zeros(256, x.size()[1], dtype=torch.float32)
        one_hot_x.scatter_(0, x.to(torch.int64), 1.0)

        # x.view()

        return one_hot_x, y.long() # one_hot_y


if __name__ == "__main__":
    # receptive_field:71, output_length:128
    # audio_files:['audio.wav.npy']
    # len(dataset):74957
    # y_pred:torch.Size([32, 256, 128]), y:torch.Size([32, 1, 128]), x:torch.Size([32, 256, 198])

    dataset = WaveNetDataset(71, 128)
    print(f"len(dataset):{len(dataset)}")
   
    for i in range(len(dataset)):
        x, y = dataset[i]
        # x.view
        print(f"x:{x.shape} , y:{y.shape}")
    #     # indices = torch.argmax(x, dim=0)
    #     # print(indices)
    #     # print(torch.max(x[0,:]))
    #     indices = torch.argmax(x, dim=0) 
    #     print(torch.max(indices))
    #     df = pd.DataFrame(indices.numpy())
    #     csv_file = f"test/{i}.csv"
    #     df.to_csv(csv_file, index=False, index_label = False)
        # print(indices)

    #     if a == 20:
    #         # df = pd.DataFrame(x.numpy())
    #         plt.stem(indices)
    #         plt.show()
    #         # Save the DataFrame to a CSV file
    #         # csv_file = 'tensor_table.csv'
    #         # df.to_csv(csv_file, index=False)
            
            
    #     a += 1
    
    # a = torch.tensor([[1,2,3,4,5,6]])
    # print(a)
    # print(a.view(1,-1))