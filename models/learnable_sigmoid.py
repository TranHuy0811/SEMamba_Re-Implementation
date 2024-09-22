import torch.nn as nn
import torch


class LearnableSigmoid1D(nn.Module):
    def __init__(self,in_feature,beta=1):
        super(LearnableSigmoid1D,self).__init__()
        self.beta=beta
        self.param=nn.Parameter(torch.ones(in_feature))
        self.param.requires_grad=True
        
    def forward(self,x):
        return self.beta*torch.sigmoid(self.param*x)



class LearnableSigmoid2D(nn.Module):
    def __init__(self,in_feature,beta=1):
        super(LearnableSigmoid2D,self).__init__()
        self.beta=beta
        self.param=nn.Parameter(torch.ones(in_feature,1))
        self.param.requires_grad=True
        
    def forward(self,x):
        return self.beta*torch.sigmoid(self.param*x)