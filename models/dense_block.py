import torch.nn as nn
import torch

def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2), 
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))



class DenseBlock(nn.Module):
    def __init__(self,cfg,kernel_size=(3,3),depth=4):
        super(DenseBlock,self).__init__()
        self.cfg=cfg
        self.hid_feature=cfg['model_config']['hid_feature']
        self.depth=depth
        self.dense_block=nn.ModuleList()
        
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.hid_feature * (i + 1), self.hid_feature, kernel_size, 
                          dilation=(dil, 1), padding=get_padding_2d(kernel_size, (dil, 1))),
                nn.InstanceNorm2d(self.hid_feature, affine=True),
                nn.PReLU(self.hid_feature)
            )
            self.dense_block.append(dense_conv)
            
    def forward(self,x):
        skip=x
        for i in range(self.depth):
            x=self.dense_block[i](skip)
            skip=torch.cat([x,skip],dim=1)
        return x