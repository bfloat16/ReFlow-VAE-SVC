import os
import numpy as np
import librosa
import torch
import argparse
from tools import utils
from tqdm import tqdm
from tools.utils import traverse_dir
import torch.multiprocessing as mp

def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default=r'./configs/reflow-vae-wavenet.yaml')
    parser.add_argument("-n", "--num_processes", type=int, default=20)
    return parser.parse_args(args=args, namespace=namespace)
    
def preprocess(id, path, filelist, sample_rate, num_processes, time_dict):
    path_srcdir = os.path.join(path, 'audio')
    file_chunk = filelist[id::num_processes]

    for file in tqdm(file_chunk):
        path_srcfile = os.path.join(path_srcdir, file)
        
        duration = librosa.get_duration(path=path_srcfile, sr=sample_rate)

        time_dict[file] = duration

def worker_init_fn(path, extensions, sample_rate, num_processes):
    print('Loading audio clips from :', path)
    time_dict = mp.Manager().dict()
    filelist = traverse_dir(os.path.join(path, 'audio'), extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
    mp.spawn(preprocess, args=(path, filelist, sample_rate, num_processes, time_dict), nprocs=num_processes)
    path_time_dict = os.path.join(path, 'time_dict.npy')
    np.save(path_time_dict, dict(time_dict))

if __name__ == '__main__':
    cmd = parse_args()
    num_processes = cmd.num_processes

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
    args = utils.load_config_yaml(cmd.config)
    train_path = args.data.train_path
    valid_path = args.data.valid_path
    extensions = args.data.extensions

    sample_rate = args.data.sampling_rate

    worker_init_fn(train_path, extensions, sample_rate, num_processes)
    worker_init_fn(valid_path, extensions, sample_rate, num_processes)