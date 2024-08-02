from hparams import HParams

class Model_Params(HParams):
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

    model_name: str = "wavenet"