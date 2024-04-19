import librosa
from model_utils import load_model
from analysis import analyze_segments
import csv
from label_groups import label_groups, special_labels

# Constants
model_url = 'https://www.kaggle.com/models/google/yamnet/tensorFlow2/yamnet/1'
audio_path = 'amadeus-bark.wav'

# Load model and audio
model = load_model(model_url)
audio, sampling_rate = librosa.load(audio_path, sr=16000)

# Load class names
class_names = []
with open('yamnet_class_map.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)  # Skip header
    for row in reader:
        class_names.append(row[2])

# Analyze audio
intervals = librosa.effects.split(audio, top_db=40)  # Example threshold
target_length = 16000  # One second of audio, defined in main.py
# Define thresholds
min_rms_threshold = 0.01
min_pitch_prob_threshold = 0.005
results = analyze_segments(audio, intervals, sampling_rate, class_names, label_groups, special_labels, model, target_length, min_rms_threshold, min_pitch_prob_threshold)


# Output results
#print(results)

for start_time, end_time, category, score in results:
    print(f"Segment: Start {start_time:.2f}s, End {end_time:.2f}s, Category: {category}, Score: {score:.4f}")
