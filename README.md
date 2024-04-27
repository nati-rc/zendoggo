# ZenDoggo

## Overview
ZenDoggo helps dog owners train their pups for alone time by analyzing audio recordings. 
It leverages machine learning for sound analysis and uses the Gemini API to deliver insights into your dog's behavior 
– helping you understand and address potential separation anxiety, even when you can't be glued to the camera 24/7

## Features
- Robust sound detection: Leverages librosa to identify segments of audio containing sound, using customizable thresholds.
- Advanced audio classification: Employs Google's YAMNet model to categorize sounds (barking, whining, human speech, etc.).
- AI-powered reports: Integrates with the Gemini Generative AI API to provide in-depth behavioral insights.
- Accessible design: User-friendly frontend for easy audio uploads and results visualization.

## Architecture
### Backend
- main.py: The heart of the Flask application, defining the routes and application logic.
- model_utils.py: Utilities for loading models from TensorFlow Hub.
- analysis.py: Contains the core functions for analyzing audio, performing sound segmentation and classification.
- audio_utils.py: Helper functions for audio feature extraction like RMS and pitch.
- label_groups.py: Definitions of label groups used for categorizing sounds.

### API Routes
- GET /: Serves the front-end application.
- POST /analyze: The critical endpoint, receiving audio files and orchestrating the analysis process, ultimately returning the results.

### Front End
- Script.js: Houses the JavaScript code responsible for uploading audio, interacting with the API, and dynamically displaying the results.
This file is located in the 'static' directory.
- Styles.css: CSS stylesheets for visually styling the frontend interface. 
This file is located in the 'static' directory.
- index.html: The structural backbone of the frontend, defining the layout and placeholders for dynamic content.
This file is located in the 'templates' directory.

## Setup and Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install required Python packages:
   ```` python
   import tensorflow as tf
   import tensorflow_hub as hub
   import librosa
   import numpy as np
   from flask import Flask, request, Response, jsonify, render_template
   import io
   import csv
   import json
   import google.generativeai as genai
   from collections import defaultdict
   import os
   ````
3. Configure environment variables: 

   Create a config.json file with your Gemini API key
   ```` json
   {
     "gemini_api_key": "your_key_here"
   }
   ````
4. Run the application locally
   ```` python
   python main.py
   ````
This will start the Flask server on http://localhost:5000/ or http://127.0.0.1:5000/, 
where you can upload audio files and receive the analysis results.

## Other Notes
- The yamnet_class_map.csv file is included for convenience. 
The latest version can be found on the official TensorFlow Models repository: https://github.com/tensorflow/models/blob/master/research/audioset/yamnet/yamnet_class_map.csv
- TensorFlow Hub: The environment variable TFHUB_MODEL_LOAD_FORMAT is set to "UNCOMPRESSED" to instruct TensorFlow Hub to fetch models directly from remote storage, potentially improving efficiency.

