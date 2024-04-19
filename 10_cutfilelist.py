import os
from tools.utils import traverse_dir

path = 'data/底模/train'
num_processes = 3
extensions = ["wav", "mp3", "ogg", "flac", "opus", "snd"]
filelist = traverse_dir(os.path.join(path, 'audio'), extensions=extensions, is_pure=True, is_sort=True, is_ext=True)

for i in range(num_processes):
    files = filelist[i::num_processes]
    with open(f'filelist_{i}.txt', 'w') as f:
        for file in files:
            f.write(file + '\n')