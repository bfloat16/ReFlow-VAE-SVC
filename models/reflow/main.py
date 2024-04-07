import os
import yaml
import torch
import torch.nn as nn
import numpy as np
from .reflow import Bi_RectifiedFlow
from .wavenet import WaveNet
from .vocoder import Vocoder

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

def load_model_vocoder(model_path,):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config_file = os.path.join(os.path.split(model_path)[0], 'config.yaml')
    with open(config_file, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    
    vocoder = Vocoder(args.vocoder.type, args.vocoder.ckpt)
       
    if args.model.type == 'RectifiedFlow_VAE':
        model = Unit2Wav_VAE(
            args.data.encoder_out_channels, 
            args.model.n_spk,
            args.model.use_pitch_aug,
            vocoder.dimension,
            args.model.n_layers,
            args.model.n_chans,
            args.model.n_hidden,
            args.model.back_bone
            )
    else:
        raise ValueError(f" [x] Unknown Model: {args.model.type}")
        
    print(' [Loading] ' + model_path)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    return model, vocoder, args

class Unit2Wav_VAE(nn.Module):
    def __init__(self, n_unit, n_spk, use_pitch_aug=False, out_dims=128, n_layers=6, n_chans=512, n_hidden=256, back_bone='wavenet'):
        super().__init__()
        self.unit_embed = nn.Linear(n_unit, out_dims)
        self.f0_embed = nn.Linear(1, n_hidden)
        self.volume_embed = nn.Linear(1, out_dims)
        if use_pitch_aug:
            self.aug_shift_embed = nn.Linear(1, n_hidden, bias=False)
        else:
            self.aug_shift_embed = None
        self.n_spk = n_spk
        if n_spk is not None and n_spk > 1:
            self.spk_embed = nn.Embedding(n_spk, n_hidden)
        if back_bone == 'wavenet':
            self.reflow_model = Bi_RectifiedFlow(WaveNet(in_dims=out_dims, n_layers=n_layers, n_chans=n_chans, n_hidden=n_hidden))
        else:
            raise ValueError(f" [x] Unknown Backbone: {back_bone}")
            
    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None, aug_shift=None, vocoder=None, gt_spec=None, infer=True, return_wav=False, infer_step=10, method='euler', t_start=0.0, use_tqdm=True):
        '''
        input: 
            B x n_frames x n_unit
        return: 
            dict of B x n_frames x feat
        '''
        # condition
        cond = self.f0_embed((1+ f0 / 700).log())
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(units.device)
                    cond = cond + v * self.spk_embed(spk_id_torch - 1)
            else:
                cond = cond + self.spk_embed(spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            cond = cond + self.aug_shift_embed(aug_shift / 5)
        
        # vae mean
        x = self.unit_embed(units) + self.volume_embed(volume)
        
        # vae noise
        x += torch.randn_like(x)
        
        x = self.reflow_model(infer=infer, x_start=x, x_end=gt_spec, cond=cond, infer_step=infer_step, method='euler', use_tqdm=True)
        
        if return_wav and infer:
            return vocoder.infer(x, f0)
        else:
            return x
            
    def vae_infer(self, input_mel, input_f0, input_spk_id, output_f0, output_spk_id=None, spk_mix_dict=None, aug_shift=None, infer_step=10, method='euler'):
        source_cond = self.f0_embed((1+ input_f0 / 700).log()) + self.spk_embed(input_spk_id - 1)
        target_cond = self.f0_embed((1+ output_f0 / 700).log())
        if self.n_spk is not None and self.n_spk > 1:
            if spk_mix_dict is not None:
                for k, v in spk_mix_dict.items():
                    spk_id_torch = torch.LongTensor(np.array([[k]])).to(input_mel.device)
                    target_cond = target_cond + v * self.spk_embed(spk_id_torch - 1)
            else:
                target_cond = target_cond + self.spk_embed(output_spk_id - 1)
        if self.aug_shift_embed is not None and aug_shift is not None:
            target_cond = target_cond + self.aug_shift_embed(aug_shift / 5)

        latent = self.reflow_model(infer=True, x_end=input_mel, cond=source_cond, infer_step=infer_step, method='euler', use_tqdm=True)
        output_mel = self.reflow_model(infer=True, x_start=latent, cond=target_cond, infer_step=infer_step, method='euler', use_tqdm=True)
        return output_mel