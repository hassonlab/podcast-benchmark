import torch
from torch import nn
from mup import MuReadout

class MLPProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.in_dim = config.get("in_dim") 
        self.out_dim = config.get("out_dim", self.in_dim)
        self.hidden_dims = config.get("projection_hidden")
        self.use_layer_norm = config.get("use_layer_norm", False)
        self.final_ln = config.get("final_ln", False)
        self.final_ln_affine = config.get("final_ln_affine", False)
        self.dropout_rate = 0.1
        
        self.final_layer_norm = None
        if self.final_ln:
            self.final_layer_norm = nn.LayerNorm(self.out_dim, elementwise_affine=self.final_ln_affine)

        self.use_mup = config.get("use_mup", False)
        if self.use_mup:
            print(f"Using MuReadout with d_model={self.in_dim}, out_dim={self.out_dim}, hidden_dims={self.hidden_dims}")

        self.projectors = self._build_projector()
            
    def _build_projector(self):
        layers = []
        in_dim = self.in_dim
        act_fn = nn.ELU()

        for hidden_dim in self.hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if self.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(act_fn)
            if self.dropout_rate > 0:
                layers.append(nn.Dropout(self.dropout_rate))
            in_dim = hidden_dim
            
        layers.append(nn.Linear(in_dim, self.out_dim)  if not self.use_mup else MuReadout(in_dim, self.out_dim, output_mult=1.0))
        
        return nn.Sequential(*layers)
        
    def forward(self, x, data_info_list = None):
        orig_shape = x.shape
        
        if len(orig_shape) == 2:
            reshaped_x = x
        elif len(orig_shape) == 3:
            B, N, D = orig_shape
            reshaped_x = x.reshape(-1, D)
        elif len(orig_shape) == 4:
            B, C, N, D = orig_shape
            reshaped_x = x.reshape(-1, D)
        else:
            raise ValueError(f"Unsupported input shape: {orig_shape}. Expected 2D, 3D, or 4D tensor.")
        
        projected_x = self.projectors(reshaped_x)
        if self.final_layer_norm is not None:
            projected_x = self.final_layer_norm(projected_x)
        
        if len(orig_shape) == 2:
            return projected_x
        elif len(orig_shape) == 3:
            return projected_x.reshape(B, N, self.out_dim)
        elif len(orig_shape) == 4:
            return projected_x.reshape(B, C, N, self.out_dim)
    
def get_feature_and_target_shapes(params, task_info_dict):
    assert task_info_dict['consistent_channels'] == True, "Currently only consistent channels supported"

    C, num_seconds, num_targets = task_info_dict['num_channels'], task_info_dict['num_seconds'], task_info_dict['num_targets']
    patch_sampling_rate = task_info_dict['patch_sampling_rate'] 
    input_token_len = params.patch_size

    N = int(num_seconds * patch_sampling_rate / input_token_len)
    d_model = params.width
    if task_info_dict['target_dynamics'] == 'discrete':
        target_shape = (num_targets, )
    elif task_info_dict['target_dynamics'] == 'continuous':
        target_sampling_rate = task_info_dict['target_sampling_rate']
        target_len = int(num_seconds * target_sampling_rate)
        target_shape = (target_len, num_targets)

    feature_shape = (C, N, d_model)
    return feature_shape, target_shape    

def get_dims(params, task_info_dict):
    feature_shape, target_shape = get_feature_and_target_shapes(params, task_info_dict)
    in_dim = torch.prod(torch.tensor(feature_shape))
    out_dim = target_shape[0]   
    if params.ft_config == "flatten_linear":
        return in_dim, out_dim
    elif params.ft_config == "flatten_mlp":
        config = {}
        config["in_dim"] = in_dim
        config["out_dim"] = out_dim
        
        task_num_seconds = task_info_dict['num_seconds']
        if params.num_mlp_layers == 2:
            config['projection_hidden'] = [200]
        else: 
            config['projection_hidden'] = ([task_num_seconds * 200] + [200] * (params.num_mlp_layers - 2)) 
        config['use_mup'] = False 
        return config
    else:
        raise NotImplementedError("currently supports only flatten_linear, flatten_mlp")