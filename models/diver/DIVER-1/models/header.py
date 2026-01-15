from torch import nn 
from models.head_components import create_multi_heads_generic, getMultiMultiheadDict_fromDummy
from models.model_builders import CustomIdentity

class Header(nn.ModuleDict):
    def __init__(self, d_model, patch_size, mup):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.mup = mup
        self['main'] = CustomIdentity() 
        patchwise_head_params = self.patchwise_ssl(d_model, patch_size)
        self['org'] = create_multi_heads_generic(patchwise_head_params)

    def patchwise_ssl(self,d_model, patch_size):
        if patch_size == 500:
            from .target_representation_functions import DEFAULT_FFT_AMPLITUDE_LOG1P_CONFIG_W01_CF200, DEFAULT_TIME_DOMAIN_MSE_CONFIG,DEFAULT_MULTISCALE_STFT_AMPLITUDE_LOG1P_CONFIG__200_100__50_25_CF200
            patchwise_ssl_configs = {
                    "patchwise_time_domain": DEFAULT_TIME_DOMAIN_MSE_CONFIG,
                    "patchwise_multiscale_stft_amplitude": DEFAULT_MULTISCALE_STFT_AMPLITUDE_LOG1P_CONFIG__200_100__50_25_CF200,
                    "patchwise_fft_amplitude": DEFAULT_FFT_AMPLITUDE_LOG1P_CONFIG_W01_CF200
                }
        elif patch_size == 50:
            from .target_representation_functions import DEFAULT_FFT_AMPLITUDE_LOG1P_CONFIG_CF200, DEFAULT_TIME_DOMAIN_MSE_CONFIG
            patchwise_ssl_configs = {
                    "patchwise_time_domain": DEFAULT_TIME_DOMAIN_MSE_CONFIG,
                    "patchwise_fft_amplitude": DEFAULT_FFT_AMPLITUDE_LOG1P_CONFIG_CF200
                } 
        else:
            raise NotImplementedError("patch size except 50 , 500 is not implented yet")
        patchwise_head_params = getMultiMultiheadDict_fromDummy(patch_size, d_model, patchwise_ssl_configs)
        if self.mup:
            for key in patchwise_head_params.keys():
                patchwise_head_params[key]['use_mup'] = True
        else:
            for key in patchwise_head_params.keys():
                patchwise_head_params[key]['use_mup'] = False

        return patchwise_head_params