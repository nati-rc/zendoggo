from flask import Flask, request, Response, jsonify
import io
import librosa
from model_utils import load_model
from analysis import analyze_segments
import csv
from label_groups import label_groups, special_labels
import numpy as np
import json
import google.generativeai as genai

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

#Load GeminiAPI key
with open('config.json') as config_file:
    config = json.load(config_file)
    gemini_api_key = config['gemini_api_key']

genai.configure(api_key=gemini_api_key)

#SETTING UP GEMINI API
# Set up the generation configuration and system instruction
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192
}

safety_settings = [{
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
}, {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
}, {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
}, {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_MEDIUM_AND_ABOVE"
}]

system_instruction = "I built an app to analyze sounds, specifically determines dog barking sounds when a dog is left alone at home. The results are in json format consisting of (Start Time in seconds , End Time in seconds, Category). I am only sharing the segments that had sound, silent segments are excluded.\n\nGenerate a summary analyzing the audio content, highlighting any interesting patterns or insights you can identify, include the percent of time the dog barked, vs. silence, vs, other noise . Give suggestion for decreasing the barking based on the analysis done, if the dog doesn't bark or barks a few then just give complimentary remarks and ways to maintain the good behavior. \n\nReturn your results in JSON format:\nPercentage Distribution: (1-2 lines)\nSummary: (small paragraph)\nSuggestions:  (3-4 bullet points)\n\n"

model_genai = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              system_instruction=system_instruction,
                              safety_settings=safety_settings)

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
    audio, sampling_rate = librosa.load(audio_file_in_memory, sr=16000)  # Load directly with desired sample rate

    # Define thresholds and target length for audio processing
    min_rms_threshold = 0.01
    min_pitch_prob_threshold = 0.005
    target_length = 16000  # One second of audio

    intervals = librosa.effects.split(audio, top_db=40)  # Example threshold
    results = analyze_segments(audio, intervals, sampling_rate, class_names, label_groups, special_labels, model, target_length, min_rms_threshold, min_pitch_prob_threshold)
    results = json.dumps(results, cls=CustomJSONEncoder)

    #Gemini API Response
    convo = model_genai.start_chat(history=[])
    convo.send_message(results)
    gemini_response = convo.last.text  # Get the response from Gemini

    return Response(gemini_response, mimetype='application/json')  # Return analysis results in json format

if __name__ == '__main__':
    app.run(debug=True)  # Turn off debug in production
