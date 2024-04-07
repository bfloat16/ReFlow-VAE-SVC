import os
import torch
from tools.utils import load_config_yaml
from models.nsf_hifigan.nvSTFT import STFT
from models.nsf_hifigan.models import load_model,load_config
from torchaudio.transforms import Resample
from models.fish.fish import Firefly

class Vocoder:
    def __init__(self, vocoder_type, vocoder_ckpt):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        if vocoder_type == 'nsf-hifigan':
            self.vocoder = NsfHifiGAN(vocoder_ckpt)
        elif vocoder_type == 'nsf-hifigan-log10':
            self.vocoder = NsfHifiGANLog10(vocoder_ckpt)
        elif vocoder_type == 'fish':
            self.vocoder = Firefly(vocoder_ckpt)
        else:
            raise ValueError(f" [Error] Unknown vocoder: {vocoder_type}")
            
        self.resample_kernel = {}
        self.vocoder_sample_rate = self.vocoder.sample_rate()
        self.vocoder_hop_size = self.vocoder.hop_size()
        self.dimension = self.vocoder.dimension()
        
    def extract(self, audio, sample_rate=0, keyshift=0):
        if sample_rate == self.vocoder_sample_rate or sample_rate == 0:
            audio_res = audio
        else:
            key_str = str(sample_rate)
            if key_str not in self.resample_kernel:
                self.resample_kernel[key_str] = Resample(sample_rate, self.vocoder_sample_rate, lowpass_filter_width = 128).to(self.device)
            audio_res = self.resample_kernel[key_str](audio)    
        
        # extract
        mel = self.vocoder.extract(audio_res, keyshift=keyshift) # B, n_frames, bins
        return mel
   
    def infer(self, mel, f0):
        f0 = f0[:,:mel.size(1),0] # B, n_frames
        audio = self.vocoder(mel, f0)
        return audio

class FireFlyGAN(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device

        self.config_path = os.path.join(os.path.split(model_path)[0], 'config.yaml')
        self.model_path = model_path
        config = load_config_yaml(self.config_path)
        if str(config.model) == 'fire_fly_gan_base_20240407':
            self.model = Firefly(self.model_path)
        else:
            raise ValueError(f" [x] Unknown model: {config.model}")

        self.model.eval()
        self.model.to(self.device)
        self.sr = config.sampling_rate
        self.hop_size = config.hop_size
        self.dim = config.num_mels
        self.stft = STFT(
            self.sr,
            self.dim,
            config.n_fft,
            config.win_size,
            config.hop_size,
            config.fmin,
            config.fmax
        )

    def sample_rate(self):
        return self.sr

    def hop_size(self):
        return self.hop_size

    def dimension(self):
        return self.dim

    def extract(self, audio, keyshift=0):
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2)  # B, n_frames, bins
        return mel

    def forward(self, mel, f0=None):
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c)
        return audio
    
class NsfHifiGAN(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.model = None
        self.h = load_config(model_path)
        self.stft = STFT(
                self.h.sampling_rate, 
                self.h.num_mels, 
                self.h.n_fft, 
                self.h.win_size, 
                self.h.hop_size, 
                self.h.fmin, 
                self.h.fmax)
    
    def sample_rate(self):
        return self.h.sampling_rate
        
    def hop_size(self):
        return self.h.hop_size
    
    def dimension(self):
        return self.h.num_mels
        
    def extract(self, audio, keyshift=0):       
        mel = self.stft.get_mel(audio, keyshift=keyshift).transpose(1, 2) # B, n_frames, bins
        return mel
    
    def forward(self, mel, f0):
        if self.model is None:
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio

class NsfHifiGANLog10(NsfHifiGAN):    
    def forward(self, mel, f0):
        if self.model is None:
            self.model, self.h = load_model(self.model_path, device=self.device)
        with torch.no_grad():
            c = 0.434294 * mel.transpose(1, 2)
            audio = self.model(c, f0)
            return audio