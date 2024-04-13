import torch

# 加载模型字典
model_path = r"\\wsl.localhost\Ubuntu-22.04\home\bfloat16\ReFlow-VAE-SVC\exp\reflowvae-wavenet\model_new_325000.pt"
state_dict = torch.load(model_path)

# 打印所有键的名称
for key in state_dict.keys():
    print(key)

print(state_dict['global_step'])

for key_ in state_dict['model'].keys():
    print(key_)

# 删除不需要的键
del state_dict['model']['spk_embed.weight']
state_dict['global_step'] = 0

torch.save(state_dict, model_path.replace('model', 'model_new'))