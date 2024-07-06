from unittest import TestCase
from torch import nn
import numpy as np
from einops import repeat
import unittest
import torch
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from wavenet import WaveNet

class TestWaveNet(TestCase):

    def setUp(self) -> None:
        super().setUp()

        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("MPS backend is available.")
        elif torch.cuda.is_available():
            print(f"CUDA is available, devices count {print(torch.cuda.device_count())}")
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            print("MPS and CUDA is not available. Using CPU instead.")

        in_channels = 1
        res_channels = 3
        skip_channels = 4
        out_channels = 5
        sample_size = 20000
        batch_size = 8

        self.wavenet = WaveNet(in_channels, res_channels, skip_channels, out_channels, kernel_size=2, stack_size=4, layer_size=10)
        self.input = nn.Parameter(torch.randn((in_channels, sample_size)))
        self.input = repeat(self.input, 'c t -> b c t', b=batch_size) # c: channel, t: time
        
        # self.wavenet.to(device)
        # self.wavenet.stack_res_block.to(torch.device("cuda"))
        # self.input = self.input.to(device)

        # print(f"WaveNet model is on device: {next(self.wavenet.parameters()).device}")
        # print(f"stack_res_block model is on device: {next(self.wavenet.stack_res_block.parameters()).device}")
        # print(f"Input is on device: {self.input.device}")

        # self.input = torch.tensor([[[1,2,3,4,5,6,7,8]]], dtype=torch.float32)
        print(f"input:{self.input.shape}, receptive_field:{self.wavenet.receptive_field}")
        
    def test_runWaveNet(self):
        output = self.wavenet(self.input)
        print(f"output:{output.shape}")

if __name__ == '__main__':
    unittest.main()
  