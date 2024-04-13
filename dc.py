import os
import glob
import wave
import numpy as np
import shutil
from tqdm import tqdm

def read_audio(path):
    with wave.open(path, 'rb') as wav_file:
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16)
        return audio_samples

def calculate_dc_offset(audio_samples):
    return np.mean(audio_samples)

def find_large_dc_offset_files(folder_path, threshold=1.0):
    audio_files = glob.glob(f'{folder_path}/**/*.wav', recursive=True)
    large_dc_offset_files = []

    for file_path in tqdm(audio_files):
        audio_samples = read_audio(file_path)
        dc_offset = calculate_dc_offset(audio_samples)
        
        if abs(dc_offset) > threshold:
            large_dc_offset_files.append((file_path, dc_offset))
    
    return large_dc_offset_files

def move_file_replace_part(src_file, base_folder, replace_from, replace_to):
    # Modify part of the path
    part_to_modify = os.path.relpath(src_file, base_folder)
    new_part = part_to_modify.replace(replace_from, replace_to)
    target_path = os.path.join(base_folder, new_part)
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.move(src_file, target_path)
    return target_path

# Example usage
base_folder = r"data/train"
folder_path = os.path.join(base_folder, "audio")
target_subfolder = "large_dc_offset"

large_dc_files = find_large_dc_offset_files(folder_path)
for file, offset in large_dc_files:
    print(f'File {file} has a large DC offset of {offset:.2f}')
    moved_path = move_file_replace_part(file, base_folder, "audio", target_subfolder)
    print(f'Moved to {moved_path}')