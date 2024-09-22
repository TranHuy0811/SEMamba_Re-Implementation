import torch.nn as nn
import torch
import einops
from .encoder_module import EncoderModule
from .mamba_block import TF_Mamba
from .decoder_module import MagDecoder,PhaDecoder

class SEMamba_Advanced(nn.Module):
    def __init__(self,cfg):
        super(SEMamba_Advanced,self).__init__()
        self.cfg=cfg
        self.num_tfmamba_blocks = cfg['model_config']['num_tfmamba'] if cfg['model_config']['num_tfmamba'] is not None else 4
        
        self.encoder_module = EncoderModule(cfg)
        self.tf_mamba = nn.ModuleList([TF_Mamba(cfg) for _ in range(self.num_tfmamba_blocks)])

        self.mag_decoder = MagDecoder(cfg)
        self.pha_decoder = PhaDecoder(cfg)
    
    
    def forward(self,noisy_mag,noisy_pha):
        '''
        Initial args:
        - noisy_mag (torch.Tensor): Noisy magnitude shape [B, F, T].
        - noisy_pha (torch.Tensor): Noisy phase shape [B, F, T].
        '''
        
        # einops.rearrange similar to torch.permute
        noisy_mag=einops.rearrange(noisy_mag,'b f t -> b t f').unsqueeze(1) # [B, 1, T, F]
        noisy_pha=einops.rearrange(noisy_pha,'b f t -> b t f').unsqueeze(1) # [B, 1, T, F]
        x = torch.cat((noisy_mag,noisy_pha),dim=1) # [B, 2, T, F]
        
        
        x = self.encoder_module(x) # Encode input
        
        for block in self.tf_mamba:
            x = block(x)
        
        denoised_mag=einops.rearrange(self.mag_decoder(x) * noisy_mag, 'b c t f -> b f t c').squeeze(-1)
        denoised_pha=einops.rearrange(self.pha_decoder(x), 'b c t f -> b f t c').squeeze(-1)
        
        denoised_com=torch.stack((denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha))
                                 , dim=-1)
        
        return denoised_mag,denoised_pha,denoised_com