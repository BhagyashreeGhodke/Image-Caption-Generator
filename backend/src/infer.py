import numpy as np
from pickle import load
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from utils import generate_caption  # moved from train_caption.py

# Load tokenizer, model, and max_length
tokenizer = load(open('models/tokenizer.pkl', 'rb'))
model = load_model('models/model_cap.h5')
max_length = load(open("models/max_length.pkl", "rb"))

def encode_image(img_path):
    model_cnn = InceptionV3(weights='imagenet')
    model_cnn = Model(model_cnn.input, model_cnn.layers[-2].output)
    img = load_img(img_path, target_size=(299, 299))
    arr = img_to_array(img)
    arr = np.expand_dims(arr, 0)
    arr = preprocess_input(arr)
    feature = model_cnn.predict(arr, verbose=0)
    return feature

# Inference
photo = encode_image('example.jpg')
caption = generate_caption(model, tokenizer, photo[0], max_length)
print("Caption:", caption)
