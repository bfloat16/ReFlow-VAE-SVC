import librosa
import scipy.signal
import soundfile as sf
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def get_filelist(in_dir):
    extensions = ['wav', 'ogg', 'opus', 'snd', 'flac']
    files = []
    for ext in extensions:
        files.extend(glob(f"{in_dir}/**/*.{ext}", recursive=True))
    return files

def process(filelist):
    target_sr = 44100
    for file_path in tqdm(filelist):
        audio, sr = librosa.load(file_path, sr=None, mono=False)

        if audio.ndim > 1:
            audio = librosa.to_mono(audio)

        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        
        b, a = scipy.signal.butter(N=5, Wn=20, btype='highpass', fs=sr)
        audio_filtered = scipy.signal.filtfilt(b, a, audio)

        sf.write(file_path, audio_filtered, sr, subtype='PCM_16')

if __name__ == '__main__':
    in_dir = r'data/底模/train/audio'
    num_processes = 20

    filelist = get_filelist(in_dir)
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        tasks = [executor.submit(process, filelist[rank::num_processes]) for rank in range(num_processes)]
        for task in tasks:
            task.result()