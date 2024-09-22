import torch.nn as nn
import torch
from functools import partial
from mamba_ssm.modules.mamba_simple import Mamba, Block
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.ops.triton.layernorm import RMSNorm



def Create_Block(
    d_model, cfg, layer_idx=0, rms_norm=True, fused_add_norm=False, residual_in_fp32=False, 
    ):
    d_state = cfg['model_config']['d_state'] # 16
    d_conv = cfg['model_config']['d_conv'] # 4
    expand = cfg['model_config']['expand'] # 4
    norm_epsilon = cfg['model_config']['norm_epsilon'] # 0.00001

    mixer_cls = partial(Mamba, layer_idx=layer_idx, d_state=d_state, d_conv=d_conv, expand=expand)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon
    )
    block = Block(
            d_model,
            mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            )
    block.layer_idx = layer_idx
    return block


class BiMambaBlock(nn.Module):
    def __init__(self,cfg,n_layer=1):
        super(BiMambaBlock,self).__init__()
        self.cfg=cfg
        self.in_channels=cfg['model_config']['hid_feature']
        
        self.forward_blocks=nn.ModuleList(Create_Block(self.in_channels,self.cfg) for i in range(n_layer))
        self.backward_blocks=nn.ModuleList(Create_Block(self.in_channels,self.cfg) for i in range(n_layer))

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
            )
        )
    

    def forward(self, x):
        x_forward,x_backward = x.clone(),torch.flip(x, [1])
        resi_forward,resi_backward = None,None

        # Forward
        for layer in self.forward_blocks:
            x_forward,resi_forward = layer(x_forward, resi_forward)
        y_forward = (x_forward + resi_forward) if resi_forward is not None else x_forward

        # Backward
        for layer in self.backward_blocks:
            x_backward,resi_backward = layer(x_backward, resi_backward)
        y_backward = torch.flip((x_backward + resi_backward),[1]) if resi_backward is not None else torch.flip(x_backward, [1])

        return torch.cat([y_forward, y_backward],-1)


    
class TF_Mamba(nn.Module):
    def __init__(self,cfg):
        super(TF_Mamba,self).__init__()
        self.cfg=cfg
        self.hid_feature=cfg['model_config']['hid_feature']
        
        self.time_mamba=BiMambaBlock(cfg,n_layer=1)
        self.freq_mamba=BiMambaBlock(cfg,n_layer=1)
        
        self.time_conv=nn.ConvTranspose1d(self.hid_feature*2,self.hid_feature,1, stride=1)
        self.freq_conv=nn.ConvTranspose1d(self.hid_feature*2,self.hid_feature,1, stride=1)
        
        
    def forward(self,x):
        # Initial args : x with shape (batch, channels, time, freq)
    
        b, c, t, f = x.size()
        
        x = x.permute(0,3,2,1).contiguous().view(b*f,t,c) 
        x = self.time_conv( self.time_mamba(x).permute(0,2,1) ).permute(0,2,1) + x 
        x = x.view(b,f,t,c).permute(0,2,1,3).contiguous().view(b*t,f,c)
        x = self.freq_conv( self.freq_mamba(x).permute(0,2,1) ).permute(0,2,1) + x
        x = x.view(b,t,f,c).permute(0,3,1,2)
        
        return x