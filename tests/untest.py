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
        in_channels = 1
        skip_channels = 1
        out_channels = 5
        sample_size = 20000
        batch_size = 8

        self.wavenet = WaveNet(in_channels, skip_channels, out_channels, kernel_size=2, stack_size=4, layer_size=10)

        # if torch.backends.mps.is_available():
        #     device = torch.device("mps")
        #     print("MPS backend is available.")
        # else:
        #     device = torch.device("cpu")
        #     print("MPS backend is not available. Using CPU instead.")

        self.input = nn.Parameter(torch.randn((in_channels, sample_size)))
        self.input = repeat(self.input, 'c t -> b c t', b=batch_size) # c: channel, t: time
        
        # self.wavenet.to(device)
        # self.input = self.input.to(device)

        # model_device = next(self.wavenet.parameters()).device
        # print(f"WaveNet model is on device: {model_device}")
        # print(f"Input is on device: {self.input.device}")

        # self.input = torch.tensor([[[1,2,3,4,5,6,7,8]]], dtype=torch.float32)
        print(f"input:{self.input.shape}, receptive_field:{self.wavenet.receptive_field}")
        
    def test_runWaveNet(self):
        output = self.wavenet(self.input)
        print(f"output:{output.shape}")

if __name__ == '__main__':
    unittest.main()
  