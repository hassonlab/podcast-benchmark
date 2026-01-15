import torch
from torch import nn
from models.layers import OriginalMoiraiEncoder 
from models.embedder import Embedder
from models.special_token import TokenManager
from models.header import CustomIdentity, Header

class DIVER(nn.Module):
   def __init__(self, d_model: int = 512, e_layer : int = 12, mup : bool = False, patch_size: int = 500):
      super().__init__() 

      self.precision = torch.float32
      self.patcher = Patchify(patch_len=patch_size, stride=patch_size)      

      self.mask_generator = PatchedRandomMasking(0.5)
      self.mask_dim = patch_size 
      self.mask_encoding = nn.Parameter(torch.zeros(self.mask_dim), requires_grad=False) 
      
      self.embedding = Embedder(d_model, e_layer, mup, patch_size)
      self.encoder = OriginalMoiraiEncoder(d_model=d_model, e_layers=e_layer, mup=mup, dropout=0.1)
      self.head = CustomIdentity()

      self.token_manager = TokenManager(d_model)
      self.heads = Header(d_model, patch_size, mup)
   
   def forward(self, x, data_info_list=None, x_mark=None, y_mark=None, use_mask=None,
               return_embedding_output = False, return_encoder_output = False, return_prepend_outputs = True):
      x = x.reshape(x.shape[0], x.shape[1], -1)
      return_dict = {}      

      x = self.patcher(x) ; return_dict['x_patched'] = x.clone() 
      
      if use_mask : 
         x, mask = self.apply_mask_patched(x, self.mask_generator)
         x = self.embedding(x, data_info_list=data_info_list) ; return_dict['embedding_output'] = x.clone() if return_embedding_output else None
      else:
         x = self.embedding(x, data_info_list=data_info_list) ; return_dict['embedding_output'] = x.clone() if return_embedding_output else None
         mask = None
      
      x = self.token_manager.prepend_x_with_tokens(x, mode ="prepend_NC", n_token_name="N_token", c_token_name="C_token", nc_token_name="NC_token",
                                                   expand_n_along_dimensions=True, expand_c_along_dimensions=True)
      x = self.encoder(x, data_info_list=data_info_list, x_mark = x_mark, y_mark = y_mark) ; return_dict['encoder_output'] = x.clone() if return_encoder_output else None
      
      token_manager_output_dict = self.token_manager.extract_prepended_x(x)
      if return_prepend_outputs:
         return_dict['token_manager_output'] = token_manager_output_dict
         x = token_manager_output_dict["org_x_position_features"]

      x = self.head(x, data_info_list=data_info_list)
      return_dict['y'] = x
      return_dict['mask'] = mask

      token_outputs = return_dict['token_manager_output']
      return_dict['y_org'] = self.heads['org'](token_outputs['org_x_position_features'], data_info_list = data_info_list)
      
      return return_dict

   def apply_mask_patched(self, x, mask_generator):
      B, C, N, P_or_D = x.shape  
      device = x.device
      mask = mask_generator(B, C, N, device=device, x=x).to(self.precision)

      masked_x = x.clone() 
      masked_x[mask == 1] = self.mask_encoding 
      return masked_x, mask


class Patchify(nn.Module):
    def __init__(self, patch_len, stride, lr_padding = (0,0), padding_method = None):
        super(Patchify, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.lr_padding = lr_padding 
        self.padding_method = padding_method  
        
    def forward(self, x, data_info_list = None):
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) 
        return x

class PatchedRandomMasking(nn.Module):
    def __init__(self, mask_ratio):
        super().__init__()
        self.mask_ratio = mask_ratio
    
    def forward(self, bz, ch_num, patch_num, device, x):
        mask = torch.zeros((bz, ch_num, patch_num), dtype=torch.long, device=device)
        mask = mask.bernoulli_(self.mask_ratio)
        return mask
    