import numpy as np
import redis
import os
from concurrent.futures import ProcessPoolExecutor

def store_numpy_to_redis(file_list):
    r = redis.Redis(host='localhost', port=6379, db=0)
    for file_path in file_list:
        array = np.load(file_path)
        array_bytes = array.tobytes()
        key = 'numpy:' + os.path.basename(file_path).replace('.npy', '')
        r.set(key, array_bytes)

def parallel_process(folder_path, num_processes):
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.npy')]
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        num_files_per_process = len(file_list) // num_processes
        tasks = [executor.submit(store_numpy_to_redis, file_list[i:i + num_files_per_process]) for i in range(0, len(file_list), num_files_per_process)]
        for future in tasks:
            future.result()

if __name__ == '__main__':
    parent_folder_path = '/path/to/your/parent/folder'
    num_processes = 4
    parallel_process(parent_folder_path, num_processes)