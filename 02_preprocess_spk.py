import os
import yaml

# 文件和文件夹路径
folder_path = r'data/train/audio'
yaml_file_path = r'configs/reflow-vae-wavenet.yaml'

# 读取子文件夹名称并排序
subfolders = [f.name for f in os.scandir(folder_path) if f.is_dir()]
sorted_subfolders = sorted(subfolders)

# 读取yaml文件
with open(yaml_file_path, 'r') as file:
    data = yaml.safe_load(file) or {}

spk_dict = {}
for i, folder in enumerate(sorted_subfolders):
    spk_dict[folder] = i

# 将更新的spk_dict保存回data字典
data['spk_dict'] = spk_dict

# 重新写入更新后的data到yaml文件
with open(yaml_file_path, 'w') as file:
    yaml.dump(data, file, allow_unicode=True)