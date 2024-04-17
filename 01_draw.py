from pydub import AudioSegment
import os
import shutil
import random

def move(src_dir, dest_dir, min_duration=3, limit=1, dir_limit=50):
    all_dirs = [os.path.join(root, d) for root, dirs, _ in os.walk(src_dir) for d in dirs]
    selected_dirs = random.sample(all_dirs, min(len(all_dirs), dir_limit))
    
    for root, dirs, files in os.walk(src_dir):
        if root not in selected_dirs:
            continue

        audio_files = []
        for file in files:
            try:
                audio_path = os.path.join(root, file)
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0  # Convert to seconds
                if duration >= min_duration:
                    audio_files.append((audio_path, duration))
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        if not audio_files:
            continue

        # Sort by duration and pick the shortest `limit` number of files
        audio_files.sort(key=lambda x: x[1])
        selected_files = audio_files[:limit]

        for audio_path, _ in selected_files:
            relative_path = os.path.relpath(audio_path, src_dir)
            dest_path = os.path.join(dest_dir, relative_path)
            dest_folder = os.path.dirname(dest_path)
            os.makedirs(dest_folder, exist_ok=True)
            shutil.move(audio_path, dest_path)
            print(f"Moved {audio_path} to {dest_path}")

if __name__ == "__main__":
    source_directory = "data/底模/train/audio"
    destination_directory = "data/底模/valid/audio"
    move(source_directory, destination_directory)