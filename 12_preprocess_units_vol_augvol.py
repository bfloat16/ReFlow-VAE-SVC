import os
import numpy as np
import random
import scipy
import librosa
import torch
import argparse
from tools import utils
from tqdm import tqdm
from models.reflow.extractors import Volume_Extractor, Units_Encoder
from tools.utils import traverse_dir
import torch.multiprocessing as mp

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=r'./configs/reflow-vae-wavenet.yaml')
    parser.add_argument("-n", "--num_processes", type=int, default=4)
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(id, path, filelist, device, encoder_type, encoder_ckpt, encoder_sample_rate, encoder_hop_size, extensions, sample_rate, hop_size, num_processes):
    path_srcdir  = os.path.join(path, 'audio')
    path_unitsdir  = os.path.join(path, 'units')
    path_volumedir  = os.path.join(path, 'volume')

    file_chunk = filelist[id::num_processes]
    
    volume_extractor = Volume_Extractor(hop_size)
    units_encoder = Units_Encoder(encoder_type, encoder_ckpt, encoder_sample_rate, encoder_hop_size)    

    filelist =  traverse_dir(path_srcdir, extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
     
    for file in tqdm(file_chunk):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)

        audio, sr = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        b, a = scipy.signal.butter(N=5, Wn=10, btype='highpass', fs=sr)
        audio = scipy.signal.filtfilt(b, a, audio)

        audio = np.ascontiguousarray(audio)

        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)

        volume = volume_extractor.extract(audio)

        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy()
       
        os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
        np.save(path_unitsfile, units.astype(np.float16))
        os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
        np.save(path_volumefile, volume)
                
if __name__ == '__main__':
    cmd = parse_args()
    num_processes = cmd.num_processes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args = utils.load_config_yaml(cmd.config)
    path = args.data.train_path
    sample_rate = args.data.sampling_rate
    hop_size = args.data.block_size
    extensions = args.data.extensions

    encoder_type = args.data.encoder
    encoder_ckpt = args.data.encoder_ckpt
    encoder_sample_rate = args.data.encoder_sample_rate
    encoder_hop_size = args.data.encoder_hop_size
    
    print('Loading audio clips from :', path)
    filelist = traverse_dir(os.path.join(path, 'audio'), extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
    mp.spawn(preprocess, args=(path, filelist, device, encoder_type, encoder_ckpt, encoder_sample_rate, encoder_hop_size, extensions, sample_rate, hop_size, num_processes), nprocs=num_processes, join=True)