import torch.nn as nn
import torch
import einops
from .dense_block import DenseBlock
from .learnable_sigmoid import LearnableSigmoid2D


class MagDecoder(nn.Module):
    def __init__(self,cfg):
        super(MagDecoder,self).__init__()
        self.cfg=cfg
        self.hid_feature=cfg['model_config']['hid_feature']
        self.output_channel=cfg['model_config']['output_channel']
        self.beta = cfg['model_config']['beta']
        self.n_fft = cfg['audio_n_stft_config']['n_fft']
        
        
        self.dense_block=DenseBlock(cfg,depth=4)
        
        self.mag_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.Conv2d(self.hid_feature, self.output_channel, (1, 1)),
            nn.InstanceNorm2d(self.output_channel, affine=True),
            nn.PReLU(self.output_channel),
            nn.Conv2d(self.output_channel, self.output_channel, (1, 1))
        )
        
        self.lsigmoid = LearnableSigmoid2D(self.n_fft // 2 + 1, beta=self.beta)
        
        
    def forward(self,x): 
        x = self.dense_block(x)
        x = self.mag_conv(x)
        x = einops.rearrange(x,'b c t f -> b f t c').squeeze(-1)
        x = self.lsigmoid(x)
        x = einops.rearrange(x,'b f t -> b t f').unsqueeze(1)
        return x
    




class PhaDecoder(nn.Module):
    def __init__(self,cfg):
        super(PhaDecoder,self).__init__()
        self.cfg=cfg
        self.hid_feature=cfg['model_config']['hid_feature']
        self.output_channel=cfg['model_config']['output_channel']
        
        
        self.dense_block=DenseBlock(cfg,depth=4)
        
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(self.hid_feature, self.hid_feature, (1, 3), stride=(1, 2)),
            nn.InstanceNorm2d(self.hid_feature, affine=True),
            nn.PReLU(self.hid_feature)
        )
        
        self.phase_conv_r = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(self.hid_feature, self.output_channel, (1, 1))
        
    def forward(self,x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i,x_r)
        return x
    