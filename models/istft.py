import torch

def Inverse_STFT(mag, pha, n_fft, hop_size, win_size, compress_factor=1.0, center=True):
    mag = torch.pow(mag, 1.0 / compress_factor)
    com = torch.complex(mag * torch.cos(pha), mag * torch.sin(pha))
    hann_window = torch.hann_window(win_size).to(com.device)
    wav = torch.istft(
                    com, 
                    n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window, 
                    center=center)
    return wav