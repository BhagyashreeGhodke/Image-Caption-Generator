import os
import pickle
import gdown # Import the gdown library
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template, redirect, url_for

# --- Define Paths based on your folder structure (Robust Method) ---
# Get the absolute path of the directory where this script is located (frontend/src)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the absolute path to the models directory
MODELS_DIR = os.path.abspath(os.path.join(script_dir, '..', '..', 'backend', 'models'))


# Create the models directory if it doesn't exist
if not os.path.exists(MODELS_DIR):
    print(f"Creating models directory at: {MODELS_DIR}")
    os.makedirs(MODELS_DIR)

# --- Google Drive File Download ---
file_ids = {
    "best_model.h5": "17_bnFfjYkcQdQMnui6N35iRp9gS7aBK-",
    "tokenizer.pkl": "1fBCUXbaH286ikeEbhmVnkrt_wmwqY8Jd",
    "max_length.pkl": "1LNUYDStnU4l-9ySkaTeBzuYN05R-qHMy"
}

def download_files_from_gdrive(files, destination_folder):
    """Checks for files and downloads them to the specified folder."""
    for filename, file_id in files.items():
        file_path = os.path.join(destination_folder, filename)
        if not os.path.exists(file_path):
            print(f"'{file_path}' not found. Downloading from Google Drive...")
            try:
                url = f'https://drive.google.com/uc?id={file_id}'
                # Tell gdown to save the file in the correct folder
                gdown.download(url, file_path, quiet=False)
                print(f"'{filename}' downloaded successfully to '{destination_folder}'.")
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
                print("Please ensure the File ID is correct and the file is shared with 'Anyone with the link'.")
        else:
            print(f"'{filename}' already exists in '{destination_folder}'. Skipping download.")

# Run the download function when the app starts
download_files_from_gdrive(file_ids, MODELS_DIR)


# --- Model and Tokenizer Loading ---
print("\n--- Loading model and tokenizer ---")
# Construct full paths to the model files
tokenizer_path = os.path.join(MODELS_DIR, 'tokenizer.pkl')
max_length_path = os.path.join(MODELS_DIR, 'max_length.pkl')
model_weights_path = os.path.join(MODELS_DIR, 'best_model.h5')

# Load the tokenizer
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Load max_length
with open(max_length_path, 'rb') as f:
    MAX_LENGTH = pickle.load(f)

# --- Configuration ---
# FIX: Hardcode the VOCAB_SIZE to match the trained model's embedding layer
VOCAB_SIZE = 8485
EMBEDDING_DIM = 256
UNITS = 512
FEATURES_SHAPE = 2048
ATTENTION_FEATURES_SHAPE = 64

# --- Keras Model Definitions ---
class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))
    score = self.V(attention_hidden_layer)
    attention_weights = tf.nn.softmax(score, axis=1)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)
    return context_vector, attention_weights

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)
    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)
    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))


# --- Load Models ---
print("Loading InceptionV3 for feature extraction...")
image_model = InceptionV3(include_top=False, weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output
image_features_extract_model = Model(new_input, hidden_layer)
print("InceptionV3 loaded.")

print("Loading captioning model...")
encoder = CNN_Encoder(EMBEDDING_DIM)
decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, VOCAB_SIZE)

# We need to run a dummy forward pass to build the models before loading weights
dummy_img_features = tf.random.uniform((64, ATTENTION_FEATURES_SHAPE, FEATURES_SHAPE))
dummy_input = tf.random.uniform((64, 1), minval=0, maxval=VOCAB_SIZE, dtype=tf.int32)
dummy_hidden = decoder.reset_state(batch_size=64)
_ = encoder(dummy_img_features)
_, _, _ = decoder(dummy_input, dummy_img_features, dummy_hidden)

# Load the trained weights from your file
decoder.load_weights(model_weights_path)
print("Captioning model loaded successfully.")


# --- Image Preprocessing and Caption Generation ---
def load_image(image_path):
    img = load_img(image_path, target_size=(299, 299))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def evaluate(image):
    attention_plot = np.zeros((MAX_LENGTH, ATTENTION_FEATURES_SHAPE))
    hidden = decoder.reset_state(batch_size=1)
    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))
    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(MAX_LENGTH):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)
        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()
        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        
        if tokenizer.index_word[predicted_id] == '<end>':
            return ' '.join(result)
            
        result.append(tokenizer.index_word[predicted_id])
        dec_input = tf.expand_dims([predicted_id], 0)

    return ' '.join(result)


# --- Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    caption = ''
    image_path = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            caption = evaluate(filepath).capitalize()
            image_path = filepath
            
    return render_template('index.html', caption=caption, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
