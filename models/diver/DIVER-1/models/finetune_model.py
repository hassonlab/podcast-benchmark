import torch
from torch import nn 
from mup import MuReadout
from utils import mup_utils
from models.diver import DIVER
from models.model_builders import MakeModelIgnoreDataInfoList, CustomIdentity
from models.finetune_builders import MLPProjector, get_dims

class FineTuneModel(nn.Module):
    def __init__(self, params, task_info_dict): 
        super().__init__()
        self.task_info_dict = task_info_dict 
        base_shape_save_dir = params.model_dir
        width = params.width
        depth = params.depth
        mup = params.mup_weights
        patch_size = params.patch_size
        
        with torch.no_grad():
            self.backbone = DIVER(d_model=width, e_layer=depth, mup=mup, patch_size=patch_size)
            if params.mup_weights:
                def _backbone_builder(w: int, d: int):
                    self.backbone= DIVER(d_model=w, e_layer=d, mup=True, patch_size=patch_size)
                    return self.backbone
                identifier="DIVER_iEEG_FINAL_model" 
                if patch_size==50:
                    identifier += "_patch50"
                mup_utils.apply_mup(target_module=self.backbone, model_builder=_backbone_builder, 
                                    identifier=identifier, width=width, depth=depth,
                                    save_dir=base_shape_save_dir)

        self.feature_extraction_func = lambda x, data_info_list=None : x['token_manager_output']['org_x_position_features'] 
        self.ft_model_input_adapter = CustomIdentity()
        self.ft_core_model = CustomIdentity()
        self.ft_model_output_adapter = CustomIdentity()

    def load_backbone_checkpoint(self, foundation_dir, device='cpu', deepspeed_pth_format=False):
        from utils import checkpoint
        checkpoint.load_model_checkpoint(
                    self.backbone, device, foundation_dir, deepspeed_pth_format=deepspeed_pth_format
        )

    def forward(self, x, data_info_list=None):
        x_features = self.backbone(x, data_info_list=data_info_list, use_mask=False, return_encoder_output=True)
        x_features = self.feature_extraction_func(x_features, data_info_list=data_info_list) 
        finetune_model_input = self.ft_model_input_adapter(x_features, data_info_list=data_info_list)
        finetune_model_output = self.ft_core_model(finetune_model_input, data_info_list=data_info_list)
        finetune_model_output = self.ft_model_output_adapter(finetune_model_output, data_info_list=data_info_list)

        return finetune_model_output

class flatten_linear_finetune(FineTuneModel):
    def __init__(self, params, task_info_dict):
        super().__init__(params, task_info_dict)
        use_mup = params.ft_mup
        in_dim, out_dim = get_dims(params, task_info_dict)

        self.ft_model_input_adapter = MakeModelIgnoreDataInfoList(nn.Flatten())
        self.ft_core_model = MakeModelIgnoreDataInfoList(nn.Linear(in_dim, out_dim) if not use_mup
            else MuReadout(in_dim, out_dim, output_mult=1.0))
        self.ft_model_output_adapter = CustomIdentity()
        
        if params.foundation_dir: 
            self.load_backbone_checkpoint(params.foundation_dir, device='cpu', deepspeed_pth_format=params.deepspeed_pth_format)

class flatten_mlp_finetune(FineTuneModel):
    def __init__(self, params, task_info_dict):
        super().__init__(params, task_info_dict)
        configs = get_dims(params, task_info_dict)
 
        self.ft_model_input_adapter = MakeModelIgnoreDataInfoList(nn.Flatten())
        self.ft_core_model = MLPProjector(configs)
        self.ft_model_output_adapter = CustomIdentity()
        
        if params.foundation_dir: 
            self.load_backbone_checkpoint(params.foundation_dir, device='cpu', deepspeed_pth_format=params.deepspeed_pth_format)