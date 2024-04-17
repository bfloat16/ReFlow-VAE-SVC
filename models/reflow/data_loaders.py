import os
import random
import numpy as np
import torch
import random
import librosa
from tqdm import tqdm
from tools.utils import traverse_dir
from torch.utils.data import Dataset

def get_data_loaders(args, whole_audio=False):
    data_train = AudioDataset(
        args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        spk_dict=args.z_spk_dict,
        load_all_data=args.train.cache_all_data,
        whole_audio=whole_audio,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk,
        device=args.train.cache_device,
        fp16=args.train.cache_fp16,
        use_aug=True
        )
    loader_train = torch.utils.data.DataLoader(
        data_train,
        batch_size=args.train.batch_size if not whole_audio else 1,
        shuffle=True,
        num_workers=args.train.num_workers if args.train.cache_device=='cpu' else 0,
        persistent_workers=(args.train.num_workers > 0) if args.train.cache_device=='cpu' else False,
        pin_memory=True if args.train.cache_device=='cpu' else False
        )
    data_valid = AudioDataset(
        args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.data.block_size,
        sample_rate=args.data.sampling_rate,
        spk_dict=args.z_spk_dict,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        n_spk=args.model.n_spk
        )
    loader_valid = torch.utils.data.DataLoader(
        data_valid,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
        )
    return loader_train, loader_valid 

class AudioDataset(Dataset):
    def __init__(self, path_root, waveform_sec, hop_size, sample_rate, spk_dict, load_all_data=True, whole_audio=False, extensions=['wav'], n_spk=1, device='cpu', fp16=False, use_aug=False):
        super().__init__()
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.paths = traverse_dir(os.path.join(path_root, 'audio'), extensions=extensions, is_pure=True, is_sort=True, is_ext=True)
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.data_buffer={}
        self.pitch_aug_dict = np.load(os.path.join(self.path_root, 'pitch_aug_dict.npy'), allow_pickle=True).item()
        self.spk_dict = spk_dict
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume, mel data from :', path_root)
            
        for name_ext in tqdm(self.paths, total=len(self.paths)):
            path_audio = os.path.join(self.path_root, 'audio', name_ext)
            duration = librosa.get_duration(path=path_audio, sr=self.sample_rate)
            
            path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
            f0 = np.load(path_f0)
            f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(device)
                
            path_volume = os.path.join(self.path_root, 'volume', name_ext) + '.npy'
            volume = np.load(path_volume)
            volume = torch.from_numpy(volume).float().unsqueeze(-1).to(device)
            
            path_augvol = os.path.join(self.path_root, 'aug_vol', name_ext) + '.npy'
            aug_vol = np.load(path_augvol)
            aug_vol = torch.from_numpy(aug_vol).float().unsqueeze(-1).to(device)

            path_mel = os.path.join(self.path_root, 'mel', name_ext) + '.npy'
            mel = np.load(path_mel)
            mel = torch.from_numpy(mel).to(device)
            
            spk_id = self.spk_dict.get(os.path.dirname(name_ext))
            if spk_id > n_spk:
                raise ValueError(f" [x] spk_id {spk_id} is larger than n_spk {n_spk}")
            spk_id = torch.LongTensor(np.array([spk_id])).to(device)
            
            if load_all_data:
                path_augmel = os.path.join(self.path_root, 'aug_mel', name_ext) + '.npy'
                aug_mel = np.load(path_augmel)
                aug_mel = torch.from_numpy(aug_mel).to(device)
                
                path_units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
                units = np.load(path_units)
                units = torch.from_numpy(units).to(device)
                
                if fp16:
                    mel = mel.half()
                    aug_mel = aug_mel.half()
                    units = units.half()
                    
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'mel': mel,
                        'aug_mel': aug_mel,
                        'units': units,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id
                        }
            else:
                self.data_buffer[name_ext] = {
                        'duration': duration,
                        'f0': f0,
                        'volume': volume,
                        'aug_vol': aug_vol,
                        'spk_id': spk_id,
                        'mel': mel
                        }
           
    def __getitem__(self, file_idx):
        name_ext = self.paths[file_idx]
        data_buffer = self.data_buffer[name_ext]
        if data_buffer['duration'] < (self.waveform_sec + 0.1):
            return self.__getitem__((file_idx + 1) % len(self.paths))
        return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        name = os.path.splitext(name_ext)[0]
        frame_resolution = self.hop_size / self.sample_rate
        duration = data_buffer['duration']
        waveform_sec = duration if self.whole_audio else self.waveform_sec
        
        # load audio
        idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
        start_frame = int(idx_from / frame_resolution)
        units_frame_len = int(waveform_sec / frame_resolution)
        aug_flag = random.choice([True, False]) and self.use_aug
        # load mel
        mel_key = 'aug_mel' if aug_flag else 'mel'
        mel = data_buffer.get(mel_key)
        if mel is None:
            mel = os.path.join(self.path_root, mel_key, name_ext) + '.npy'
            mel = np.load(mel)
            mel = mel[start_frame : start_frame + units_frame_len]
            mel = torch.from_numpy(mel).float() 
        else:
            mel = mel[start_frame : start_frame + units_frame_len]
            
        # load units
        units = data_buffer.get('units')
        if units is None:
            units = os.path.join(self.path_root, 'units', name_ext) + '.npy'
            units = np.load(units)
            units = units[start_frame : start_frame + units_frame_len]
            units = torch.from_numpy(units).float() 
        else:
            units = units[start_frame : start_frame + units_frame_len]

        # load f0
        f0 = data_buffer.get('f0')
        aug_shift = 0
        if aug_flag:
            aug_shift = self.pitch_aug_dict[name_ext]
        f0_frames = 2 ** (aug_shift / 12) * f0[start_frame : start_frame + units_frame_len]
        
        # load volume
        vol_key = 'aug_vol' if aug_flag else 'volume'
        volume = data_buffer.get(vol_key)
        volume_frames = volume[start_frame : start_frame + units_frame_len]
        
        # load spk_id
        spk_id = data_buffer.get('spk_id')
        
        # load shift
        aug_shift = torch.from_numpy(np.array([[aug_shift]])).float()
        
        return dict(mel=mel, f0=f0_frames, volume=volume_frames, units=units, spk_id=spk_id, aug_shift=aug_shift, name=name, name_ext=name_ext)

    def __len__(self):
        return len(self.paths)