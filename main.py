from flask import Flask, request, Response, jsonify
import io
import librosa
from model_utils import load_model
from analysis import analyze_segments
import csv
from label_groups import label_groups, special_labels
import numpy as np
import json

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert numpy arrays to lists
        return super().default(obj)

app = Flask(__name__)
# Constants
model_url = 'https://www.kaggle.com/models/google/yamnet/tensorFlow2/yamnet/1'

# Load model
model = load_model(model_url)

# Load class names
class_names = []
with open('yamnet_class_map.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)  # Skip header
    for row in reader:
        class_names.append(row[2])

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'Missing audio file'}), 400

    audio_file = request.files['audio_file']
    audio_file_in_memory = io.BytesIO(audio_file.read())
    audio, sr = librosa.load(audio_file_in_memory, sr=16000)  # Load directly with desired sample rate

    # Define thresholds and target length for audio processing
    min_rms_threshold = 0.01
    min_pitch_prob_threshold = 0.005
    target_length = 16000  # One second of audio

    intervals = librosa.effects.split(audio, top_db=40)  # Example threshold
    results = analyze_segments(audio, intervals, sr, class_names, label_groups, special_labels, model, target_length, min_rms_threshold, min_pitch_prob_threshold)
    results = json.dumps(results, cls=CustomJSONEncoder)

    return Response(results, mimetype='application/json')  # Return analysis results in json format

if __name__ == '__main__':
    app.run(debug=True)  # Turn off debug in production
