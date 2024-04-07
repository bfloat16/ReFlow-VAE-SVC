import os
import numpy as np
import random
import librosa
import torch
import argparse
import shutil
from tools import utils
from tqdm import tqdm
from models.reflow.extractors import F0_Extractor
from models.reflow.vocoder import Vocoder
from tools.utils import traverse_dir
import torch.multiprocessing as mp

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=r'./configs/reflow-vae-wavenet.yaml')
    parser.add_argument("-n", "--num_processes", type=int, default=8)
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(id, path, filelist, device, f0_extractor_type, sample_rate, hop_size, f0_max, f0_min, vocoder_type, vocoder_ckpt, use_pitch_aug, num_processes, pitch_aug_dict):
    path_srcdir  = os.path.join(path, 'audio')
    path_f0dir  = os.path.join(path, 'f0')
    path_meldir  = os.path.join(path, 'mel')
    path_augmeldir  = os.path.join(path, 'aug_mel')
    path_skipdir = os.path.join(path, 'skip')

    file_chunk = filelist[id::num_processes]
    
    f0_extractor = F0_Extractor(f0_extractor_type, sample_rate, hop_size, f0_min, f0_max)
    mel_extractor = Vocoder(vocoder_type, vocoder_ckpt)
    if mel_extractor.vocoder_sample_rate != sample_rate or mel_extractor.vocoder_hop_size != hop_size:
        print('Error: Unmatch vocoder parameters')
        return None

    for file in tqdm(file_chunk):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_f0file = os.path.join(path_f0dir, binfile)
        path_melfile = os.path.join(path_meldir, binfile)
        path_augmelfile = os.path.join(path_augmeldir, binfile)
        path_skipfile = os.path.join(path_skipdir, file)
        
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        
        f0 = f0_extractor.extract(audio, uv_interp = False)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            audio_t = torch.from_numpy(audio).float().to(device)
            audio_t = audio_t.unsqueeze(0)
        
            mel_t = mel_extractor.extract(audio_t, sample_rate)
            mel = mel_t.squeeze().to('cpu').numpy()
            
            max_amp = float(torch.max(torch.abs(audio_t))) + 1e-5
            max_shift = min(1, np.log10(1/max_amp))
            log10_vol_shift = random.uniform(-1, max_shift)
            if use_pitch_aug:
                keyshift = random.uniform(-5, 5)
            else:
                keyshift = 0
            
            aug_mel_t = mel_extractor.extract(audio_t * (10 ** log10_vol_shift), sample_rate, keyshift = keyshift)
            aug_mel = aug_mel_t.squeeze().to('cpu').numpy()

            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
            os.makedirs(os.path.dirname(path_melfile), exist_ok=True)
            np.save(path_melfile, mel)
            os.makedirs(os.path.dirname(path_augmelfile), exist_ok=True)
            np.save(path_augmelfile, aug_mel)
            pitch_aug_dict[file] = keyshift
        else:
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
            os.makedirs(os.path.dirname(path_skipfile), exist_ok=True)
            shutil.move(path_srcfile, os.path.dirname(path_skipfile))

if __name__ == '__main__':
    cmd = parse_args()
    num_processes = cmd.num_processes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    args = utils.load_config_yaml(cmd.config)
    path = args.data.train_path
    extensions = args.data.extensions

    f0_extractor_type = args.data.f0_extractor
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    f0_max = args.data.f0_max
    f0_min = args.data.f0_min

    vocoder_type = args.vocoder.type
    vocoder_ckpt = args.vocoder.ckpt

    use_pitch_aug = args.model.use_pitch_aug

    pitch_aug_dict = mp.Manager().dict()

    filelist = traverse_dir(os.path.join(path, 'audio'), extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
    mp.spawn(preprocess, args=(path, filelist, device, f0_extractor_type, sample_rate, hop_size, f0_max, f0_min, vocoder_type, vocoder_ckpt, use_pitch_aug, num_processes, pitch_aug_dict), nprocs=num_processes)

    path_pitchaugdict = os.path.join(path, 'pitch_aug_dict.npy')
    np.save(path_pitchaugdict, dict(pitch_aug_dict))