import os
import yaml

# 文件和文件夹路径
folder_path = r'data/底模/train/audio'
yaml_file_path = r'configs/reflow-vae-wavenet-底模.yaml'

# 读取子文件夹名称并排序
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
sorted_subfolders = sorted(subfolders)

# 读取yaml文件
with open(yaml_file_path, 'r', encoding='utf-8') as file:
    data = yaml.safe_load(file) or {}

spk_dict = {}
for i, folder in enumerate(sorted_subfolders):
    spk_dict[folder] = i

# 将更新的spk_dict保存回data字典
data['z_spk_dict'] = spk_dict

# 更新model中的n_spk值
num_spk = len(sorted_subfolders)
data['model']['n_spk'] = num_spk

# 重新写入更新后的data到yaml文件
with open(yaml_file_path, 'w', encoding='utf-8') as file:
    yaml.dump(data, file, allow_unicode=True)