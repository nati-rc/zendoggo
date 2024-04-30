from flask import Flask, request, Response, jsonify, render_template
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
model_url = 'https://tfhub.dev/google/yamnet/1'

# Load model
model = load_model(model_url)

#Load GeminiAPI key that is located on config.json
with open('config.json') as config_file:
    config = json.load(config_file)
    gemini_api_key = config['gemini_api_key']

genai.configure(api_key=gemini_api_key)

# Set Up of Gemini API along with configuration and system instructions
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    #"response_mime_type": "application/json",
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

system_instruction = "I built an app to analyze sounds, specifically determines dog barking sounds when a dog is left alone at home. The results are in json format consisting of (Start Time in seconds , End Time in seconds, Category). I am only sharing the segments that had sound, silent segments are excluded.\n\nHere is a short description of the categories:\nDog: Canine vocalizations including barks, howls, whines, growls, and panting.\nCat: Feline sounds including purrs, meows, hisses, and caterwauls.\nBird: Chirps, squawks, songs, and wing flapping associated with birds.\nLivestock and Wildlife: Sounds from farm animals (cows, pigs, etc.), wild animals (roars, howls), and smaller creatures like rodents.\nHuman Speech: Spoken words, conversations, and vocal expressions like narration.\nHuman Sound: Non-speech vocalizations like shouts, laughter, crying, coughs, sneezes, and other bodily sounds.\nMusical Instruments and Sounds: Sounds produced by musical instruments, singing, and different musical genres.\nNature: Natural sounds like animals (birds, insects, frogs), water, wind, and weather events.\nVehicle: Sounds from cars, trucks, trains, aircraft, and other modes of transportation.\nHousehold Sound: Common sounds found in homes, including appliances, doors, and everyday objects.\nOutdoor Sound: Ambient noises characteristic of outdoor environments, both rural and urban.\nSilence: The absence of detectable sound.\nHuman Activity: Non-vocal human sounds like footsteps, clapping, chewing, and bodily functions.\nMusic: Organized sounds with melody, rhythm, and often vocals, spanning various genres and styles.\nWeather: Sounds associated with weather phenomena like wind, rain, and thunderstorms.\nConstruction: Sounds of tools, machinery, and activities characteristic of construction sites.\n\nGenerate a summary analyzing the audio content, highlighting any interesting patterns or insights you can identify, include the percent of time the dog barked, vs. silence, vs, other noise . Give suggestion for decreasing the barking based on the analysis done, if the dog doesn't bark or barks a few then just give complimentary remarks and ways to maintain the good behavior. Don't forget to mention to encourage users to consult a veterinarian for professional evaluation.\n\nFor the summary consider the following questions to help with the analysis:\n\"Were there any long stretches of continuous barking?\"\n\"Did the dog's barking coincide with any other specific sounds?\"\n\"Were there any periods where the barking seemed particularly intense?\"\n\nFor the suggestions consider mentioning to look at other signs of separation anxiety such as:\nDestructive behavior: Chewing, scratching, etc.\nElimination issues: Urinating or defecating indoors.\nPacing, restlessness: Inability to settle.\n\nThreshold for separation anxiety.  If barking exceeds 20% of the audio with no clear triggers, separation anxiety becomes more likely. This applies to audio segments that are 15min or longer. For shorter segments you can still use this but give the disclaimer that the audio is too short and is better to at least have a 15min audio.\n\n\nReturn your results in JSON format without any other markdown:\nPercentage Distribution: (1-2 lines)\nSummary: (small paragraph)\nSuggestions:  (3-4 bullet points)\n\n"

model_genai = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                              generation_config=generation_config,
                              system_instruction=system_instruction,
                              safety_settings=safety_settings)

# Loading class names that will be used for the sound categorization.
# "yamnet_class_map.csv" is included in repository but latest file can be found in
# https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
class_names = []
with open('yamnet_class_map.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    next(reader)  # Skip header
    for row in reader:
        class_names.append(row[2])

# GET /: Serves the front-end application
@app.route('/')
def index():
    return render_template('index.html')

#POST /analyze: Endpoint for uploading audio files and receiving analysis.
@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'Missing audio file'}), 400

    audio_file = request.files['audio_file']
    audio_file_in_memory = io.BytesIO(audio_file.read())
    audio, sampling_rate = librosa.load(audio_file_in_memory, sr=16000)  # Load directly with desired sample rate

    # Define thresholds and target length for audio processing. These thresholds are used to omit low background noises
    min_rms_threshold = 0.015
    min_pitch_prob_threshold = 0.005
    target_length = 16000  # One second of audio
    intervals = librosa.effects.split(audio, top_db=35)

    # We call our main analyze function
    results = analyze_segments(audio, intervals, sampling_rate, class_names, label_groups, special_labels, model, target_length, min_rms_threshold, min_pitch_prob_threshold)
    results_json = json.dumps(results, cls=CustomJSONEncoder)

    # Gemini API Response
    convo = model_genai.start_chat(history=[])
    convo.send_message(results_json)
    gemini_response = convo.last.text

# Gemini JSON response currently includes markdown so these need to be removed to get a valid JSON
    def clean_json_response(gemini_response):
        # Check if the response starts with Markdown code block marker for JSON
        gemini_response = gemini_response.strip()  # Remove any leading/trailing whitespace
        if (gemini_response.startswith("```json")):
            gemini_response = gemini_response[len("```json"):].strip()
        if gemini_response.endswith("```"):
            gemini_response = gemini_response[:-len("```")].strip()
        return gemini_response

    clean_gemini_response = clean_json_response(gemini_response)
    dict_gemini_response = json.loads(clean_gemini_response)

    combined_json_response = {
        'analysis_results': results,
        'gemini_response': dict_gemini_response
    }
    combined_json_response = json.dumps(combined_json_response, cls=CustomJSONEncoder)
    print(combined_json_response)
    return combined_json_response

if __name__ == '__main__':
    app.run(debug=True)  # Turn off debug in production
