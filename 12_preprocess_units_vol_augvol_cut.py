import os
import numpy as np
import librosa
import torch
import argparse
from tools import utils
from tqdm import tqdm
from models.reflow.extractors import Volume_Extractor, Units_Encoder

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=r'configs/reflow-vae-wavenet-底模.yaml')
    parser.add_argument("-f", "--filelist", type=str, default=r'./filelist_0.txt')
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(path, filelist, device, sample_rate, hop_size):
    path_srcdir  = os.path.join(path, 'audio')
    path_unitsdir  = os.path.join(path, 'units')
    path_volumedir  = os.path.join(path, 'volume')  
     
    for file in tqdm(filelist):
        binfile = file + '.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_unitsfile = os.path.join(path_unitsdir, binfile)
        path_volumefile = os.path.join(path_volumedir, binfile)

        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

        audio_t = torch.from_numpy(audio).float().to(device)
        audio_t = audio_t.unsqueeze(0)

        volume = volume_extractor.extract(audio)

        units_t = units_encoder.encode(audio_t, sample_rate, hop_size)
        units = units_t.squeeze().to('cpu').numpy()
       
        os.makedirs(os.path.dirname(path_unitsfile), exist_ok=True)
        with open(path_unitsfile, 'wb') as f:
            np.save(f, units.astype(np.float16))

        os.makedirs(os.path.dirname(path_volumefile), exist_ok=True)
        with open(path_volumefile, 'wb') as f:
            np.save(f, volume)
                
if __name__ == '__main__':
    cmd = parse_args()

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

    volume_extractor = Volume_Extractor(hop_size)
    units_encoder = Units_Encoder(encoder_type, encoder_ckpt, encoder_sample_rate, encoder_hop_size)

    print('Loading audio clips from :', path)
    with open(cmd.filelist, 'r') as f:
        filelist = f.read().splitlines()

    preprocess(path, filelist, device, sample_rate, hop_size)