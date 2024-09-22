import torch

def STFT(audio, n_fft, hop_size, win_size, compress_factor=1.0, center=True, addeps=False, eps=1e-10):
    eps = 1e-10
    hann_window = torch.hann_window(win_size).to(audio.device)
    
    stft_spec = torch.stft(
                    audio, n_fft, 
                    hop_length=hop_size, 
                    win_length=win_size, 
                    window=hann_window,
                    center=center, 
                    pad_mode='reflect', 
                    normalized=False, 
                    return_complex=True)

    if addeps==False:
        mag = torch.abs(stft_spec)
        pha = torch.angle(stft_spec)
    else:
        real_part = stft_spec.real
        imag_part = stft_spec.imag
        mag = torch.sqrt(real_part.pow(2) + imag_part.pow(2) + eps)
        pha = torch.atan2(imag_part + eps, real_part + eps)
        
        
    mag = torch.pow(mag, compress_factor)
    com = torch.stack((mag * torch.cos(pha), mag * torch.sin(pha)), dim=-1)
    return mag, pha, com