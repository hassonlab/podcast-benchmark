import torch
from typing import Tuple, Union, Optional

DEFAULT_TIME_DOMAIN_MSE_CONFIG = {
    "use": True, 
    "weight": 1.0, 
    "criterion": "MSELoss"
}

DEFAULT_FFT_AMPLITUDE_LOG1P_CONFIG_CF200 = {
    "use": True, 
    "weight": 1.0, 
    "criterion": "MSELoss",
    "fft_params": {
        "loss_type": "amplitude", 
        "n_fft": 500, 
        "normalize": True,
        "compress_func": "log1p", 
        "cutofffreq": 200,
        "sr":500
    }
}

DEFAULT_FFT_AMPLITUDE_LOG1P_CONFIG_W01_CF200 = {
    "use": True, 
    "weight": 0.1, 
    "criterion": "MSELoss",
    "fft_params": {
        "loss_type": "amplitude", 
        "n_fft": 500, 
        "normalize": True,
        "compress_func": "log1p", 
        "cutofffreq": 200, 
        "sr":500
    }
}

DEFAULT_MULTISCALE_STFT_AMPLITUDE_LOG1P_CONFIG__200_100__50_25_CF200 = {
    "use": True, 
    "weight": 1.0, 
    "criterion": "MSELoss",
    "loss_type" : "amplitude",
    "compress_func" : "log1p",
    "stft_params_list": [{
        "loss_type" : "amplitude",
        "n_fft": 200, 
        "hop_length": 100, 
        "win_length": None, 
        "window": "hann",
        "center": True, 
        "pad_mode": "reflect", 
        "cutofffreq" : 200,
        "sr" : 500,
        "normalize": True, 
    },{
        "loss_type" : "amplitude",
        "n_fft": 50, 
        "hop_length": 25, 
        "win_length": None, 
        "window": "hann",
        "center": True, 
        "pad_mode": "reflect", 
        "cutofffreq" : 200,
        "sr" : 500,
        "normalize": True, 
    }]
}


def compressor_transform(x: torch.Tensor, comp_func="identity"):
    if torch.is_complex(x):
        if comp_func == "log":
            raise ValueError("Log compression is not defined for complex numbers.")
        elif comp_func == "log1p":
            ratio = torch.log1p(torch.abs(x)+1e-16) / (torch.abs(x)+1e-16)
        elif comp_func == "identity":
            ratio = torch.ones_like(x) 
        else:
            raise ValueError(f"Unknown compression function: {comp_func}")
        x = x * ratio
        return x
    else:
        if comp_func == "log":
            if torch.any(x < 0):
                raise ValueError("Input tensor contains negative values, which is not allowed for log compression.")
            x = x + 1e-16
            if torch.any(x == 0):
                raise ValueError("there is zero")
            return torch.log(x)
        elif comp_func == "log1p":
            return torch.log1p(x)
        elif comp_func == "identity":
            return x
        else:
            raise ValueError(f"Unknown compression function: {comp_func}")
        
    
def single_stft_transform(x: torch.Tensor, n_fft: int, hop_length: int, 
                   win_length: int, window: str, pad_mode : str, normalize : bool, loss_type : str, cutofffreq = None, sr = 500, center :bool =True, 
                   ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    if window is None or window == "hann":
        window = torch.hann_window(n_fft if win_length is None else win_length)
    else:
        raise NotImplementedError("only hann window is implemented now")
    if window.device != x.device:
        window = window.to(x.device) 
        
    x_detached = x.detach()
    
    x_float32 = x_detached.float()
    
    stft_result = torch.stft(
            x_float32, 
            n_fft=n_fft, 
            hop_length=hop_length, 
            win_length=win_length, 
            window=window, 
            center=False, 
            pad_mode=pad_mode, 
            normalized=normalize, 
            onesided=True, 
            return_complex=True
        )
    stft_result = stft_result.to(x.dtype)
    
    if cutofffreq != None:
        cutofffreq_index = min(int(cutofffreq // (sr/n_fft)),stft_result.shape[-2])
        stft_result_freqcut = stft_result[...,0:cutofffreq_index,:]
        return stft_result_freqcut
    else:
        return stft_result
  

def multiscale_stft_transform(x: torch.Tensor, multiscale_stft_params: list, loss_type: str = "amplitude", compress_func: str="identity"):
    if loss_type not in ["amplitude","complex","phase"]:
        raise ValueError("multiscale_stft's loss type have to be amplitude, complex, or phase")
    if len(multiscale_stft_params)>1:
        total_stft_results=[]
        for param in multiscale_stft_params:
            stft_result = single_stft_transform(x, **param)
            if loss_type == "amplitude":
                stft_result = torch.abs(stft_result)
            elif loss_type == "phase":
                stft_result = torch.angle(stft_result)
            stft_result_flattend = torch.flatten(stft_result,start_dim=-2)
            total_stft_results.append(stft_result_flattend)
        out = torch.cat(total_stft_results, dim=-1)
    elif len(multiscale_stft_params) == 1:
        stft_result = single_stft_transform(x, **multiscale_stft_params[0])
        if loss_type == "amplitude":
            stft_result = torch.abs(stft_result)
        elif loss_type == "phase":
            stft_result = torch.angle(stft_result)
        out = torch.flatten(stft_result,start_dim=-2)
    elif len(multiscale_stft_params) < 1:
        raise ValueError("you need at least one stft_param for multiscale stft params")
    if loss_type == "complex" :
        out = compressor_transform(out, comp_func=compress_func)
        if compress_func == "log":
            raise ValueError("Log compression is not defined for complex numbers.")
    elif loss_type == "amplitude":
        out = compressor_transform(out, comp_func=compress_func)
    elif loss_type == "phase":
        if compress_func != "identity":
            raise ValueError("Phase loss must use 'identity' compression function.")  
        out = compressor_transform(out, comp_func=compress_func)

    return out




def fft_transform(x: torch.Tensor, n_fft: int = 256, 
                   freqcutoff: Optional[float] = None, loss_type: str = "complex", 
                   normalize: bool = False, compress_func="identity", cutofffreq=None, sr=500) -> torch.Tensor:
    x_detached = x.detach()
    
    x_float32 = x_detached.float()
    
    fft_result = torch.fft.rfft(x_float32, n=n_fft, dim=-1, norm="backward") 
    
    fft_result = fft_result.to(x.dtype)


    if cutofffreq != None:
        cutofffreq_index = min(int(cutofffreq // (sr/n_fft)),fft_result.shape[-1])
        fft_result = fft_result[...,0:cutofffreq_index]

    
    if loss_type == "complex" :
        fft_result = compressor_transform(fft_result, comp_func=compress_func)
        if compress_func == "log":
            raise ValueError("Log compression is not defined for complex numbers.")
    elif loss_type == "amplitude":
        fft_result = torch.abs(fft_result)
        fft_result = compressor_transform(fft_result, comp_func=compress_func)
    elif loss_type == "phase":
        fft_result = torch.angle(fft_result)
        if compress_func != "identity":
            raise ValueError("Phase loss must use 'identity' compression function.")
        
    return fft_result