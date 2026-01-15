"""
STFT Preprocessor for PopT model.

This module implements STFT preprocessing matching the original PopT neuroprobe implementation.
"""

import numpy as np
import torch
import torch.nn as nn
from scipy import signal


def zscore(a, axis):
    """Z-score normalization.
    
    Args:
        a: Input array
        axis: Axis along which to compute normalization
        
    Returns:
        Normalized array
    """
    mn = a.mean(axis=axis, keepdims=True)
    std = a.std(axis=axis, ddof=0, keepdims=True)
    std[(std == 0)] = 1.0  # Handle zero std
    z = (a - mn) / std
    return z


class STFTPreprocessor(nn.Module):
    """STFT preprocessing module matching original PopT implementation.
    
    This preprocessor converts raw neural signals to spectrograms using STFT,
    matching the original PopT neuroprobe implementation.
    """
    
    def __init__(self, fs=2048, freq_channel_cutoff=40, nperseg=400, 
                 noverlap=350, normalizing='zscore'):
        """
        Initialize STFT preprocessor.
        
        Args:
            fs: Sampling frequency (Hz). Default: 2048
            freq_channel_cutoff: Number of frequency channels to keep. Default: 40
            nperseg: Length of each segment for STFT. Default: 400
            noverlap: Number of points to overlap between segments. Default: 350
            normalizing: Normalization method ('zscore' or 'db'). Default: 'zscore'
        """
        super(STFTPreprocessor, self).__init__()
        self.fs = fs
        self.freq_channel_cutoff = freq_channel_cutoff
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.normalizing = normalizing
    
    def get_stft(self, x, fs, show_fs=-1, normalizing=None, **kwargs):
        """
        Compute STFT of input signal.
        
        Args:
            x: Input signal [n_electrodes, n_timesteps] or [batch, n_electrodes, n_timesteps]
            fs: Sampling frequency
            show_fs: Number of frequency channels to keep (-1 for all)
            normalizing: Normalization method ('zscore' or 'db')
            **kwargs: Additional arguments for scipy.signal.stft
            
        Returns:
            f: Frequency array
            t: Time array
            linear: Spectrogram tensor [n_electrodes, time, freq_channels] or 
                    [batch, n_electrodes, time, freq_channels]
        """
        f, t, Zxx = signal.stft(x, fs, **kwargs)
        
        if "return_onesided" in kwargs and kwargs["return_onesided"] == True:
            if show_fs > 0:
                Zxx = Zxx[:, :show_fs]
                f = f[:show_fs]
        
        # Take absolute value
        Zxx = np.abs(Zxx)
        
        # Normalize
        if normalizing == "zscore":
            Zxx = zscore(Zxx, axis=-1)
            if (Zxx.std() == 0).any():
                Zxx = np.ones_like(Zxx)
            # Remove edge effects (original PopT removes 10 samples from each end)
            if Zxx.shape[-1] > 20:
                Zxx = Zxx[:, :, 10:-10]
        elif normalizing == "db":
            Zxx = np.log(Zxx)
        
        # Handle NaN values
        if np.isnan(Zxx).any():
            Zxx = np.nan_to_num(Zxx, nan=0.0)
        
        # Transpose: [n_electrodes, freq, time] -> [n_electrodes, time, freq]
        return f, t, torch.Tensor(np.transpose(Zxx, [0, 2, 1]))
    
    def forward(self, wav):
        """
        Forward pass: convert raw signal to spectrogram.
        
        Automatically uses GPU if input tensor is on GPU, otherwise uses CPU (scipy).
        
        Args:
            wav: Input signal
                - [n_electrodes, n_timesteps] for single sample
                - [batch, n_electrodes, n_timesteps] for batched input
                - [batch, 1, n_timesteps] for single channel batched input
                
        Returns:
            linear: Spectrogram tensor (on same device as input)
                - [n_electrodes, time, freq_channels] for single sample
                - [batch, n_electrodes, time, freq_channels] for batched input
                - [batch, 1, time, freq_channels] for single channel batched input
        """
        # Check if input is torch.Tensor and on GPU
        if isinstance(wav, torch.Tensor) and wav.is_cuda:
            # Use GPU-accelerated STFT
            return self._forward_gpu(wav)
        else:
            # Use CPU-based STFT (scipy) for numpy arrays or CPU tensors
            return self._forward_cpu(wav)
    
    def _forward_gpu(self, wav):
        """
        GPU-accelerated STFT using torch.stft.
        
        Args:
            wav: Input signal tensor on GPU
                - [n_electrodes, n_timesteps] or [batch, n_electrodes, n_timesteps]
                
        Returns:
            linear: Spectrogram tensor on GPU
                - [n_electrodes, time, freq_channels] or [batch, n_electrodes, time, freq_channels]
        """
        device = wav.device
        original_shape = wav.shape
        
        # Handle different input shapes
        if len(wav.shape) == 2:
            # [n_electrodes, n_timesteps] -> add batch dimension
            wav = wav.unsqueeze(0)
            squeeze_output = True
        else:
            # [batch, n_electrodes, n_timesteps]
            squeeze_output = False
        
        batch_size, num_electrodes, n_timesteps = wav.shape
        
        # Reshape to [batch * num_electrodes, n_timesteps] for batch processing
        wav_reshaped = wav.contiguous().view(-1, n_timesteps)  # [batch * num_electrodes, n_timesteps]
        
        # Calculate hop_length from nperseg and noverlap
        hop_length = self.nperseg - self.noverlap
        
        # Create window function on same device
        window = torch.hann_window(self.nperseg, device=device)
        
        # Compute STFT: [batch * num_electrodes, n_timesteps] -> [batch * num_electrodes, freq_bins, time_frames]
        # torch.stft expects input of shape [..., n_timesteps]
        stft_complex = torch.stft(
            wav_reshaped,
            n_fft=self.nperseg,
            hop_length=hop_length,
            win_length=self.nperseg,
            window=window,
            return_complex=True,
            normalized=False,
            onesided=True,
            center=False  # Match scipy behavior
        )  # [batch * num_electrodes, freq_bins, time_frames]
        
        # Take absolute value (magnitude)
        stft_mag = torch.abs(stft_complex)  # [batch * num_electrodes, freq_bins, time_frames]
        
        # Select frequency channels (freq_channel_cutoff)
        if self.freq_channel_cutoff > 0:
            stft_mag = stft_mag[:, :self.freq_channel_cutoff, :]  # [batch * num_electrodes, freq_channels, time_frames]
        
        # Transpose: [batch * num_electrodes, freq_channels, time_frames] -> [batch * num_electrodes, time_frames, freq_channels]
        stft_mag = stft_mag.transpose(1, 2)  # [batch * num_electrodes, time_frames, freq_channels]
        
        # Normalize
        if self.normalizing == "zscore":
            # Z-score normalization along time dimension (axis=1)
            mean = stft_mag.mean(dim=1, keepdim=True)  # [batch * num_electrodes, 1, freq_channels]
            std = stft_mag.std(dim=1, keepdim=True)  # [batch * num_electrodes, 1, freq_channels]
            std = torch.where(std == 0, torch.ones_like(std), std)  # Handle zero std
            stft_mag = (stft_mag - mean) / std
            
            # Remove edge effects (original PopT removes 10 samples from each end)
            if stft_mag.shape[1] > 20:
                stft_mag = stft_mag[:, 10:-10, :]
        elif self.normalizing == "db":
            stft_mag = torch.log(stft_mag + 1e-10)  # Add small epsilon to avoid log(0)
        
        # Handle NaN values
        stft_mag = torch.nan_to_num(stft_mag, nan=0.0)
        
        # Reshape back: [batch * num_electrodes, time_frames, freq_channels] -> [batch, num_electrodes, time_frames, freq_channels]
        stft_mag = stft_mag.view(batch_size, num_electrodes, -1, self.freq_channel_cutoff)
        
        # Remove batch dimension if input was 2D
        if squeeze_output:
            stft_mag = stft_mag.squeeze(0)  # [num_electrodes, time_frames, freq_channels]
        
        return stft_mag
    
    def _forward_cpu(self, wav):
        """
        CPU-based STFT using scipy.signal.stft (original implementation).
        
        Args:
            wav: Input signal (numpy array or CPU tensor)
                - [n_electrodes, n_timesteps] for single sample
                - [batch, n_electrodes, n_timesteps] for batched input
                
        Returns:
            linear: Spectrogram tensor on CPU
                - [n_electrodes, time, freq_channels] for single sample
                - [batch, n_electrodes, time, freq_channels] for batched input
        """
        # Handle different input shapes
        if len(wav.shape) == 2:
            # [n_electrodes, n_timesteps]
            if isinstance(wav, torch.Tensor):
                wav_np = wav.detach().cpu().numpy()
            else:
                wav_np = wav
            _, _, linear = self.get_stft(
                wav_np,
                self.fs,
                show_fs=self.freq_channel_cutoff,
                nperseg=self.nperseg,
                noverlap=self.noverlap,
                normalizing=self.normalizing,
                return_onesided=True
            )
            return linear
        elif len(wav.shape) == 3:
            # [batch, n_electrodes, n_timesteps] or [batch, 1, n_timesteps]
            batch_size = wav.shape[0]
            spec_list = []
            for b in range(batch_size):
                wav_batch = wav[b]
                if isinstance(wav_batch, torch.Tensor):
                    wav_np = wav_batch.detach().cpu().numpy()
                else:
                    wav_np = wav_batch
                _, _, linear = self.get_stft(
                    wav_np,
                    self.fs,
                    show_fs=self.freq_channel_cutoff,
                    nperseg=self.nperseg,
                    noverlap=self.noverlap,
                    normalizing=self.normalizing,
                    return_onesided=True
                )
                spec_list.append(linear)
            return torch.stack(spec_list, dim=0)
        else:
            raise ValueError(f"Unexpected input shape: {wav.shape}. Expected 2D or 3D tensor.")

