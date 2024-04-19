import torch

model_path = r"exp/reflowvae-wavenet-attention/model_155000.pt"
state_dict = torch.load(model_path)

for key in state_dict.keys():
    print(key)

print(state_dict['global_step'])
print(state_dict['model']['unit_embed.weight'].shape)

for key_ in state_dict['model'].keys():
    print(key_)

del state_dict['model']['spk_embed.weight']
state_dict['global_step'] = 0

torch.save(state_dict, model_path.replace('model', 'model_new'))