import librosa
import numpy as np

def calculate_rms(segment_audio):
    return np.mean(librosa.feature.rms(y=segment_audio)[0])

def calculate_pitch(segment_audio, sr):
    f0, _, voiced_probs = librosa.pyin(segment_audio, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=sr)
    return f0, np.nanmean(voiced_probs)  # Returning average voiced probability
