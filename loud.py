import librosa
import numpy as np
import soundfile as sf

def load_audio_and_energy(path):
    y, sr = librosa.load(path, sr=None)
    hop_length = 512
    frame_length = 2048
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2))
        for i in range(0, len(y), hop_length)
    ])
    return y, sr, energy

def exponential_moving_average(x, alpha=0.9):
    ema = [x[0]]
    for i in range(1, len(x)):
        ema.append(alpha * x[i] + (1 - alpha) * ema[-1])
    return np.array(ema)

# Load audio files
source_audio, source_sr, source_energy = load_audio_and_energy('花月夜.wav')
target_audio, target_sr, target_energy = load_audio_and_energy('花月夜_128.wav')

# Apply smoothing
source_energy_smooth = exponential_moving_average(source_energy)

# Prevent divide by zero errors
epsilon = 1e-8
target_energy += epsilon  # Add a small constant to target energy

# Calculate gain factors safely
gain_factors = np.sqrt(source_energy_smooth / target_energy)

# Adjust target audio
adjusted_target_audio = np.zeros_like(target_audio)
frame_length = 2048
hop_length = 512
for i, gain in enumerate(gain_factors):
    start = i * hop_length
    end = start + frame_length
    if end < len(target_audio):
        adjusted_target_audio[start:end] += target_audio[start:end] * gain

# Save the adjusted audio
sf.write('adjusted_target_audio.wav', adjusted_target_audio, target_sr)