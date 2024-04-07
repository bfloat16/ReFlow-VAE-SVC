import os
import yaml

folder_path = r'data/train/audio'
yaml_file_path = r'configs/reflow-vae-wavenet.yaml'

subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]

sorted_subfolders = sorted(subfolders)

with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file) or {}

spk_dict = data.get('spk_dict', {})

start_index = len(spk_dict) + 1
for i, folder in enumerate(sorted_subfolders, start=start_index):
    spk_dict[folder] = i

data['spk_dict'] = spk_dict

with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, allow_unicode=True)