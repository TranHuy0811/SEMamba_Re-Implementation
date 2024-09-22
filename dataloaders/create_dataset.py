import torch.nn as nn
import torch, torchaudio
from torch.utils.data import Dataset
import random
from models.stft import STFT


class CustomDataset(Dataset):
    def __init__(self,
            text_file,
            sample_rate=16000,
            do_split=True,
            segment_size=32000,
            n_fft=400, 
            hop_size=100, 
            win_size=400,
            center=True,
            compress_factor=1.0,
            addeps=False, # True: Add a small eps value to avoid divide by zero error
            eps=1e-10,
            pcs=False
        ):
        self.txtfile=text_file
        self.sample_rate=sample_rate
        self.do_split=do_split
        self.segment_size=segment_size
        self.n_fft=n_fft
        self.hop_size=hop_size 
        self.win_size=win_size
        self.center=center
        self.addeps=addeps
        self.eps=eps
        self.compress_factor=compress_factor
        self.pcs=pcs
      
    
    def LoadAudio(self,path):
        waveform,initial_rate=torchaudio.load(path)
        if initial_rate!=self.sample_rate:
            waveform=torchaudio.transforms.Resample(orig_freq=initial_rate,new_freq=self.sample_rate)(waveform)
        return waveform
    
    def __getitem__(self,idx):
        noisy_path,clean_path=self.txtfile[idx].strip().split('\t')
        noisy_audio=self.LoadAudio(noisy_path)
        clean_audio=self.LoadAudio(clean_path)
        
        # Normalizing Audio
        norm_factor = torch.sqrt(len(noisy_audio) / torch.sum(noisy_audio ** 2.0))
        clean_audio = (clean_audio * norm_factor)
        noisy_audio = (noisy_audio * norm_factor)

        assert noisy_audio.shape==clean_audio.shape
        
        if self.do_split==True:
            # .size(1) same as .shape[1]
            if clean_audio.size(1)>=self.segment_size:
                max_audio_start=clean_audio.size(1)-self.segment_size
                random_audio_start=random.randint(0,max_audio_start)
                noisy_audio=noisy_audio[:,random_audio_start:random_audio_start+self.segment_size]
                clean_audio=clean_audio[:,random_audio_start:random_audio_start+self.segment_size]
            else:
                noisy_audio=torch.nn.functional.pad(noisy_audio,(0,self.segment_size-noisy_audio.size(1)),'constant')
                clean_audio=torch.nn.functional.pad(clean_audio,(0,self.segment_size-clean_audio.size(1)),'constant')
            
            
        noisy_mag,noisy_pha,noisy_com=STFT(noisy_audio,self.n_fft, self.hop_size, self.win_size, 
                                               self.compress_factor, self.center, self.addeps, self.eps)    
        clean_mag,clean_pha,clean_com=STFT(clean_audio,self.n_fft, self.hop_size, self.win_size, 
                                               self.compress_factor, self.center, self.addeps, self.eps)
            
            
        return clean_audio.squeeze() ,noisy_mag.squeeze(), noisy_pha.squeeze(), noisy_com.squeeze(), clean_mag.squeeze(), clean_pha.squeeze(), clean_com.squeeze()
    
    def __len__(self):
        return len(self.txtfile)

def CreateDataset(file_path,cfg,pcs,do_split):
    txt_file=open(file_path,'r',encoding='utf-8').readlines()

    sample_rate = cfg['audio_n_stft_config']['sample_rate']
    n_fft = cfg['audio_n_stft_config']['n_fft']
    hop_size = cfg['audio_n_stft_config']['hop_size'] 
    win_size = cfg['audio_n_stft_config']['win_size']
    segment_size = cfg['data_config']['segment_size']
    addeps = cfg['data_config']['addeps']
    eps = cfg['data_config']['eps']
    compress_factor = cfg['model_config']['compress_factor']

    return CustomDataset(text_file = txt_file, sample_rate = sample_rate, n_fft = n_fft, hop_size = hop_size, win_size = win_size,
                         do_split = do_split, segment_size = segment_size, addeps = addeps, eps = eps, pcs=pcs, compress_factor=compress_factor)