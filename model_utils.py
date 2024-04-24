import os
import tensorflow_hub as hub

# Direct TensorFlow Hub to use remote storage directly
os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"

def load_model(model_url):
    # Load the model from TensorFlow Hub
    model = hub.load(model_url)
    return model