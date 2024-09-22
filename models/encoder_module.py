import torch.nn as nn
import torch
from .dense_block import DenseBlock

class EncoderModule(nn.Module):
    def __init__(self,cfg):
        super(EncoderModule,self).__init__()
        self.cfg=cfg
        self.input_channel=cfg['model_config']['input_channel']
        self.hid_feature=cfg['model_config']['hid_feature']
        
        # Change to higher dimension
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(self.input_channel, self.hid_feature, (1, 1)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )
        
        self.dense_block=DenseBlock(cfg,depth=4)
        
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )

        
    def forward(self,x):
        x=self.dense_conv_1(x) # [batch, hid_feature, time, freq]
        x=self.dense_block(x)  # [batch, hid_feature, time, freq]
        x=self.dense_conv_2(x) # [batch, hid_feature, time, freq/2]
        return x