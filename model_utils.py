import tensorflow_hub as hub

def load_model(model_url):
    return hub.load(model_url)