import os
import librosa
import pyloudnorm as pyln
import numpy as np
from tqdm import tqdm

def calculate_loudness(audio_path):
    # 加载音频文件
    data, rate = librosa.load(audio_path, sr=None)  # sr=None 确保使用音频原始的采样率
    # 使用pyloudnorm库计算响度
    meter = pyln.Meter(rate)  # 创建响度计
    loudness = meter.integrated_loudness(data)
    return loudness

def average_loudness(folder_path):
    loudness_values = []
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith(".wav"):  # 假设音频文件为wav格式
            file_path = os.path.join(folder_path, filename)
            try:
                loudness = calculate_loudness(file_path)
                loudness_values.append(loudness)
            except:
                pass
    # 计算平均响度
    if loudness_values:
        return np.mean(loudness_values)
    else:
        return None

def process_folders(base_folder):
    folder_loudness = {}
    # 遍历基本文件夹中的每个子文件夹
    base_folder_ = os.listdir(base_folder)
    base_folder_.sort()
    for subfolder in base_folder_:
        path = os.path.join(base_folder, subfolder)
        if os.path.isdir(path):
            avg_loudness = average_loudness(path)
            folder_loudness[subfolder] = avg_loudness
            # 使用 f-string 格式化浮点数，保留两位小数
            print(f"{avg_loudness:.2f}: {subfolder}")

# 指定基本文件夹路径
base_folder = 'data/train/audio'
process_folders(base_folder)