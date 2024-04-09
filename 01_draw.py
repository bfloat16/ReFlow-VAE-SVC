from pydub import AudioSegment
import os
import shutil

def move(src_dir, dest_dir, min_duration=3, limit=1):
    for root, dirs, files in os.walk(src_dir):
        audio_files = []
        for file in files:
            try:
                audio_path = os.path.join(root, file)
                audio = AudioSegment.from_file(audio_path)
                duration = len(audio) / 1000.0  # 转换为秒
                if duration >= min_duration:
                    audio_files.append((audio_path, duration))
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")

        if not audio_files:
            continue

        # 按时长排序并选取时长最短的limit个文件
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
    source_directory = "data/train/audio"
    destination_directory = "data/val/audio"
    move(source_directory, destination_directory)