import torch
import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import numpy as np
from functools import partial
from typing import Tuple
from models.original_moirai_encoder import *

class Test_CNN_KernelSize_Effect_500(nn.Module):
    def __init__(self, d_model=512, stride=64, split=8, depth=3):
        super().__init__()
        self.sampling_rate = 512
        self.d_model = d_model
        self.stride = stride
        self.split = split
        self.depth = depth
        self.hidden_out_channel = int((self.d_model / self.sampling_rate) * self.stride)
        self.kernel_size = (self.sampling_rate // split) - 1
        self.padding = (self.sampling_rate // (split*2)) - 1
        self.out_size = self.sampling_rate // self.stride
        
        layers = [
            nn.Conv2d(
                in_channels=1,
                out_channels=self.hidden_out_channel,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding)
            ),
            nn.GroupNorm(self.out_size, self.hidden_out_channel),
            nn.GELU()
        ]

        for _ in range(depth-1):
            layers += [
                nn.Conv2d(
                    in_channels=self.hidden_out_channel,
                    out_channels=self.hidden_out_channel,
                    kernel_size=(1, 3),
                    stride=(1, 1),
                    padding=(0, 1)
                ),
                nn.GroupNorm(self.out_size, self.hidden_out_channel),
                nn.GELU()
            ]

        self.proj_in = nn.Sequential(*layers)

    def forward(self, x, data_info_list = None):
        x = F.pad(x, (6, 6))                        
        B, C, N, P = x.shape
        x = x.contiguous().view(B, 1, C * N, P)     
        x = self.proj_in(x)                         
        x = x.permute(0, 2, 1, 3).contiguous().view(B, C, N, self.d_model) 
        return x
    
class Test_CNN_KernelSize_Effect_50(nn.Module):
    def __init__(self, d_model=50, stride=4, split=1, depth=3):
        super().__init__()
        self.sampling_rate = 64
        self.d_model = d_model
        self.stride = stride
        self.split = split
        self.depth = depth
        self.hidden_out_channel = int((self.d_model / self.sampling_rate) * self.stride)
        self.kernel_size = (self.sampling_rate // split) - 1
        self.padding = (self.sampling_rate // (split*2)) - 1
        self.out_size = self.sampling_rate // self.stride
        
        layers = [
            nn.Conv2d(
                in_channels=1,
                out_channels=self.hidden_out_channel,
                kernel_size=(1, self.kernel_size),
                stride=(1, self.stride),
                padding=(0, self.padding)
            ),
            nn.GroupNorm(self.out_size, self.hidden_out_channel),
            nn.GELU()
        ]

        for _ in range(depth-1):
            layers += [
                nn.Conv2d(
                    in_channels=self.hidden_out_channel,
                    out_channels=self.hidden_out_channel,
                    kernel_size=(1, 3),
                    stride=(1, 1),
                    padding=(0, 1)
                ),
                nn.GroupNorm(self.out_size, self.hidden_out_channel),
                nn.GELU()
            ]

        self.proj_in = nn.Sequential(*layers)

    def forward(self, x, data_info_list = None):
        x = F.pad(x, (7, 7))                      
        B, C, N, P = x.shape
        x = x.contiguous().view(B, 1, C * N, P)    
        x = self.proj_in(x)                       
        x = x.permute(0, 2, 1, 3).contiguous().view(B, C, N, self.d_model) 
        return x

class CBraModSpectralEmbedding(nn.Module):
    def __init__(self, d_model=256, rfft_size=101, dropout=0.1):
        super().__init__()
        self.spectral_proj = nn.Sequential(
            nn.Linear(rfft_size, d_model),  
            nn.Dropout(dropout),
        )

    def forward(self, x, data_info_list = None):
        B, C, N, P = x.shape
        x_flat = x.contiguous().view(B*C*N, P)
        original_dtype = x_flat.dtype
        if x_flat.dtype in [torch.float16, torch.bfloat16]:
            x_flat = x_flat.float()
        spectral = torch.fft.rfft(x_flat, dim=-1, norm='forward')
        spectral_abs = torch.abs(spectral).contiguous().view(B, C, N, -1)
        if original_dtype in [torch.float16, torch.bfloat16]:
            spectral_abs = spectral_abs.to(original_dtype)
        return self.spectral_proj(spectral_abs)

class PositionalEncoding3D(nn.Module):
    def __init__(self, d_model):
        super(PositionalEncoding3D, self).__init__()
        self.d_model = d_model
        self.temperature = 2000 
        self.scale = 1/256
        self.pec = PositionEmbeddingCoordsSine(n_dim=3, d_model = d_model, temperature=self.temperature, scale=self.scale)
        self.projection = None

    def forward(self, x, data_info_list):
        B, C, N, D = x.shape
        if data_info_list is None:
            raise ValueError("data_info_list must be provided to determine channel positions.")
        pos = torch.stack([(v.detach() if torch.is_tensor(v) else torch.as_tensor(v)).to(device=x.device, dtype=x.dtype) for v in (info['xyz_id'] for info in data_info_list)], dim=0)
        mask = torch.isnan(pos).any(dim=-1, keepdim=True)
        pos = pos.masked_fill(mask,0.0)
        
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)

        pe = self.pec.forward(pos)
        pe = pe.masked_fill(mask,0.0)

        pos_expanded = pe.unsqueeze(2).repeat(1, 1, N, 1) 
        if pos_expanded.dtype != x.dtype:
            pos_expanded = pos_expanded.to(x.dtype)
        
        return pos_expanded

class PositionEmbeddingCoordsSine(nn.Module):
    def __init__(self, n_dim: int = 1, d_model: int = 256, temperature=10000, scale=1/256):
        super(PositionEmbeddingCoordsSine, self).__init__()

        self.n_dim = n_dim
        self.num_pos_feats = d_model // n_dim // 2 * 2
        self.temperature = temperature
        self.padding = d_model - self.num_pos_feats * self.n_dim

        if scale is None or scale =="None":
            scale = 1/256
        print(scale)
        self.scale = scale * 2.0 * math.pi

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        assert xyz.shape[-1] == self.n_dim

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=xyz.device)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        xyz = xyz * self.scale
        pos_divided = xyz.unsqueeze(-1) / dim_t
        pos_sin = pos_divided[..., 0::2].sin()
        pos_cos = pos_divided[..., 1::2].cos()
        pos_emb = torch.stack([pos_sin, pos_cos], dim=-1).reshape(*xyz.shape[:-1], -1)

        pos_emb = F.pad(pos_emb, (0, self.padding))
        return pos_emb

class ChannelTypeEmbedding(nn.Module):
    def __init__(self, channel_types, d_emb_model, learnable=True):
        super(ChannelTypeEmbedding, self).__init__()
        if channel_types is None or len(channel_types) == 0:
            raise ValueError("Warning: available channel_types is None or empty.")
        self.channel_types = channel_types
        self.d_emb = d_emb_model
        if learnable:
            self.embedding = nn.Embedding(len(channel_types), d_emb_model)
        else:
            self.register_buffer('embedding', torch.randn(len(channel_types), d_emb_model))

    def forward(self, x, data_info_list):
        B, C, N, D = x.shape
        if data_info_list is None:
            raise ValueError("data_info_list must be provided to determine channel types.")
        else:
            ch_modality_indices = torch.tensor(
                    [self.channel_types.index(info['modality']) for info in data_info_list],
                    dtype=torch.long, device=x.device
            )  
            ch_modality_indices = ch_modality_indices.unsqueeze(1).repeat(1, C)  
            emb = self.embedding(ch_modality_indices)
            emb = emb.unsqueeze(2).expand(B, C, N, self.d_emb)

        return emb
    
class ChannelSubTypeEmbedding(nn.Module):
    def __init__(self, channel_subtypes, d_emb_model, learnable=True):
        super(ChannelSubTypeEmbedding, self).__init__()
        if channel_subtypes is None or len(channel_subtypes) == 0:
            raise ValueError("Warning: available channel_subtypes is None or empty.")
        self.channel_subtypes = channel_subtypes
        self.d_emb = d_emb_model
        if learnable:
            self.embedding = nn.Embedding(len(channel_subtypes), d_emb_model)
        else:
            self.register_buffer('embedding', torch.randn(len(channel_subtypes), d_emb_model))
    
    def make_proper_coord_types(self, coord_subtype_list):
        old_coord_subtype = coord_subtype_list.copy()
        new_coord_subtype= []
        for coordtype in old_coord_subtype:
            if coordtype in ["Depth", "depth", "LID", "RDA", "RDP", "LAD", "LPD", "RFD","RPTD", "AHD", "AID", "MHD", "MID", "PID", "LTPD", "LATD", "LMTD", "PHD", "RATD"]:
                new_coord_subtype.append("depth")
            elif coordtype in ["Strip","strip","LAT", "LMT", "LPT", "LTO", "RST",  "RAT", "RMT", "RPT",  "RF",  "RTP", "LO", "ROF", "LF", "LTO",  "LOF", "LTP"]:
                new_coord_subtype.append("strip")
            elif coordtype in ["Grid","grid","GRID", "L Grid", "RFGRID", "Grid"]:
                new_coord_subtype.append("grid")
            else:
                new_coord_subtype.append("Unknown")
        return new_coord_subtype

    def forward(self, x, data_info_list=None):
        B, C, N, _ = x.shape
        if data_info_list is None:
            raise ValueError("data_info_list must be provided to determine channel types.")
        indices_list = []
        mask_list = [] 
        for sample in data_info_list:
            if sample.get('coord_subtype') is None or len(sample.get('coord_subtype')) != C:
                indices_list.append([0] * C)
                mask_list.append([True] * C)
            else:
                cur_indices = []
                cur_mask = []
                old_coord_subtype = sample.get('coord_subtype')
                new_coord_subtype = self.make_proper_coord_types(old_coord_subtype)
                for sub in new_coord_subtype:
                    if sub == "Unknown":
                        cur_indices.append(0)  
                        cur_mask.append(True) 
                    else: 
                        cur_indices.append(self.channel_subtypes.index(sub))
                        cur_mask.append(False)
                
                indices_list.append(cur_indices)
                mask_list.append(cur_mask)
    
        indices_tensor = torch.tensor(indices_list, dtype=torch.long, device=x.device)
        mask_tensor = torch.tensor(mask_list, dtype=torch.bool, device=x.device)  

        emb = self.embedding(indices_tensor)
        emb = emb.unsqueeze(2).expand(B, C, N, self.d_emb)
    
        mask_tensor = mask_tensor.unsqueeze(2).unsqueeze(3).expand(B, C, N, self.d_emb)
        emb = emb * (~mask_tensor).to(emb.dtype)
    
        return emb

class SlidingWindowTransformer(nn.Module):
    def __init__(
            self, model: nn.Module,  c_kernel_size : int = None, n_kernel_size : int = None,
            stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0), 
            normalize : bool = True,
        ):
        super().__init__()
        assert c_kernel_size is not None or n_kernel_size is not None, "Either c_kernel_size or n_kernel_size or both must be provided."
        self.c_kernel_size = c_kernel_size
        self.n_kernel_size = n_kernel_size
        self.model = model 
        self.stride = stride
        self.padding = padding
        self.normalize = normalize
        
    def _get_normalization_factor(self, x_unfolded: torch.Tensor, output_size : Tuple[int, int], fold_kwargs_dict: dict) -> torch.Tensor:
        ones = torch.ones_like(x_unfolded, dtype = x_unfolded.dtype, 
                               device = x_unfolded.device, requires_grad = False)
        overlap_count = F.fold(ones, output_size=output_size, **fold_kwargs_dict)
        return overlap_count        
    
    def _get_kernel_size(self, input_C_dim, input_N_dim) -> Tuple[int, int]:
        if self.n_kernel_size is not None:
            return (input_C_dim, self.n_kernel_size)
    
    def forward(self, x: torch.Tensor, data_info_list = None) -> torch.Tensor:
        B, C, N, D = x.shape        
        K_C, K_N = self._get_kernel_size(C, N) 
        common_fold_unfold_kwargs = {
            'kernel_size': (K_C, K_N),
            'stride': self.stride,
            'padding': self.padding,
        }
        x_reshape_for_unfold = rearrange(x, 'B C N D -> (B D) 1 C N')
        x_unfolded = F.unfold(x_reshape_for_unfold, **common_fold_unfold_kwargs)
        x_reshape_for_transformer = rearrange(x_unfolded, '(B D) (K_C K_N) N_k -> (B N_k) K_C K_N D', 
                                                     B=B, D=D, K_C = K_C, K_N = K_N)  
        
               
        x_transformed = self.model(x_reshape_for_transformer)
        x_reshape_for_folding_back = rearrange(x_transformed, '(B N_k) K_C K_N D -> (B D) (K_C K_N) N_k', 
                                                      B=B, D=D, K_C = K_C, K_N = K_N) 
        x_folded = F.fold(x_reshape_for_folding_back, output_size=(C, N), **common_fold_unfold_kwargs)
        x_final = rearrange(x_folded, '(B D) 1 C N -> B C N D', B=B, D=D)        
        
        if self.normalize:
            overlap_count = self._get_normalization_factor(x_unfolded, output_size=(C, N), fold_kwargs_dict=common_fold_unfold_kwargs)       
            reshaped_overlap_count = rearrange(overlap_count, '(B D) 1 C N -> B C N D', B=B, D=D)
            x_final = x_final / reshaped_overlap_count
        return x_final
        
class TimeWiseSTCPE(SlidingWindowTransformer):
    def __init__(self, model: nn.Module, stride: int = 1, timewindow_size: int = None,  normalize: bool = True):
        padding = (0, timewindow_size - 1)
        super().__init__(
            model=model, c_kernel_size=None, n_kernel_size=timewindow_size,
            stride=(1, stride), padding=padding, normalize=normalize
        )
        self.timewindow_size = timewindow_size
        self.padding_method = "full"


class OriginalMoiraiEncoder(nn.Module): 
    def __init__(self, d_model, e_layers,  mup, dropout=0.0):
        super().__init__()
        self.d_model = d_model
        self.e_layers = e_layers
        n_heads = d_model//32
        d_ff = int(4*d_model)
        self.activation = nn.SiLU()
        self.encoder = TransformerEncoder(d_model=self.d_model,
                                                        num_layers=self.e_layers,
                                                        num_heads=n_heads,
                                                        num_groups=n_heads,
                                                        pre_norm=True,
                                                        attn_dropout_p=dropout,
                                                        dropout_p=dropout,
                                                        norm_layer=RMSNorm, 
                                                        activation=self.activation, 
                                                        use_moe=False,
                                                        use_glu=True, 
                                                        use_qk_norm=True, 
                                                        var_attn_bias_layer=partial(BinaryAttentionBias), 
                                                        time_attn_bias_layer=None,
                                                        var_qk_proj_layer=None,
                                                        time_qk_proj_layer=partial(
                                                            QueryKeyProjection,
                                                            proj_layer=RotaryProjection,
                                                            kwargs=dict(max_len=512),
                                                            partial_factor=None,
                                                        ),
                                                        shared_var_attn_bias=False,
                                                        shared_time_qk_proj=True,
                                                        d_ff=d_ff,
                                                        use_bias=False,
                                                        flash_attention=True,
                                                        use_mup=mup,
                                                    ) 
    
    def forward(self, x, data_info_list = None, x_mark = None, y_mark = None, attn_mask=None, tau=None, delta=None):
        assert len(x.shape) == 4, f"Input shape should be [B, C, N, D], (D : model dim, analagous to P (patch_len)) but got {x.shape}"
        B, C, N, D = x.shape
        device = x.device
        x = rearrange(x, 'b c n d -> b (c n) d')

        variate_id = torch.repeat_interleave(
                    torch.arange(C, device=x.device), repeats=N).expand(B, -1)

        time_id = torch.tile(
                    torch.arange(N, device=x.device), (C,)).expand(B, -1)

        if data_info_list is not None and len(data_info_list) > 0 and 'input_was_resampled_and_padded' in data_info_list[0] and data_info_list[0]['input_was_resampled_and_padded'] == True: 
            padding_array = np.array([data_info['padding'] for data_info in data_info_list])
            padding_chs   = torch.tensor(padding_array[:, 0], dtype=torch.long, device=device).unsqueeze(1)
            padding_times = torch.tensor(padding_array[:, 1], dtype=torch.long, device=device).unsqueeze(1)

            variate_mask = variate_id.clone()
            time_mask = time_id.clone()

            variate_mask[variate_id >= padding_chs] = -1
            time_mask[time_id    >= padding_times] = -1
            attn_mask = torch.ones((B, C*N, C*N), dtype=torch.bool, device=x.device)

            q_mask       = (variate_mask != -1) & (time_mask != -1)
            kv_mask      = q_mask
            valid_mask   = q_mask.unsqueeze(2) & kv_mask.unsqueeze(1)
            attn_mask   &= valid_mask
            
        x = self.encoder(x, attn_mask, time_id=time_id, var_id=variate_id) 
        x = rearrange(x, 'b (c n) d -> b c n d', c=C, n=N)

        return x