import torch
import torch.nn as nn

class AddEmbedding(nn.Module):
    def __init__(self, embedding_module):
        super().__init__()
        self.embedding = embedding_module
    def forward(self, x, data_info_list = None):
        self.embedding(x, data_info_list)
        return x + self.embedding(x, data_info_list)
    
class ConcateEmbedding(nn.Module):
    def __init__(self, embedding_module1, embedding_module2):
        super().__init__()
        self.embedding1 = embedding_module1
        self.embedding2 = embedding_module2
        
    def forward(self, x, data_info_list):
        x1 = self.embedding1(x, data_info_list)
        x2 = self.embedding2(x, data_info_list)
        x = torch.cat([x1, x2], dim=-1)
        return x

class CombinedEmbedding(nn.Module):
    def __init__(self, *models):
        super().__init__()
        self.models = nn.ModuleList(models)
        
    def forward(self, x, data_info_list = None):
        return sum(model(x, data_info_list) for model in self.models)

class MakeModelIgnoreDataInfoList(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x, data_info_list=None):
        if x.shape == (5,19,30,256):
            print("IN MakeModelIgnoreDataInfoList")
            print(x.dtype, x.shape, self.module, 
                  self.module.__class__.__name__)
            print("self.module dtype", next(self.module.parameters()).dtype)
            
        return self.module(x)
    
class DataInfoAwareSequential(nn.Sequential):
    def forward(self, x, data_info_list=None):
        for module in self:
            x = module(x, data_info_list=data_info_list)
        return x

class CustomIdentity(nn.Module):
    def __init__(self):
        super(CustomIdentity, self).__init__()
    def forward(self, x, *args, **kwargs):
        return x
    