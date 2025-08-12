import os, string
import numpy as np
from pickle import dump, load
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import tensorflow as tf

# 1️⃣ Load captions and clean
def load_captions(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        doc = f.read().strip().split('\n')
    captions = {}
    for line in doc:
        if ',' not in line:  # ✅ CORRECT check
            continue
        image_id, caption = line.split(',', 1)
        image_id = image_id.split('.')[0]
        captions.setdefault(image_id, []).append('startseq ' + caption.lower() + ' endseq')
    print(f"✅ Loaded captions for {len(captions)} images")
    return captions



def clean_captions(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, cap in enumerate(caps):
            cap = cap.lower().translate(table)
            cap = ' '.join([w for w in cap.split() if len(w)>1 and w.isalpha()])
            caps[i] = cap

# 2️⃣ Create tokenizer on all captions
def create_tokenizer(captions):
    all_caps = [cap for caps in captions.values() for cap in caps]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(all_caps)
    return tokenizer

# 3️⃣ Extract image features using InceptionV3
def extract_features(directory):
    model = InceptionV3(weights='imagenet')
    model = Model(model.input, model.layers[-2].output)
    features = {}
    for fname in tqdm(os.listdir(directory)):
        img_id = fname.split('.')[0]
        img = load_img(os.path.join(directory, fname), target_size=(299,299))
        arr = img_to_array(img)
        arr = np.expand_dims(arr,0)
        arr = preprocess_input(arr)
        features[img_id] = model.predict(arr, verbose=0)
    return features

# 4️⃣ Data generator to yield sequences
def data_generator(captions, features, tokenizer, max_length, vocab_size):
    while True:
        for img_id, caps in captions.items():
            if img_id not in features:
                continue  # Skip if no feature
            feature = features[img_id][0]
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    yield [feature, in_seq], out_seq

# 5️⃣ Build the model
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# 6️⃣ Generate caption for new image
def generate_caption(model, tokenizer, photo, max_length):
    in_text = 'startseq'
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([photo.reshape((1,2048)), seq], verbose=0)
        word = tokenizer.index_word.get(np.argmax(yhat))
        if word is None: break
        in_text += ' ' + word
        if word == 'endseq': break
    return in_text.replace('startseq','').replace('endseq','').strip()

# 📌 Main execution
if __name__=='__main__':
    captions = load_captions('../Flickr8k.token.txt')
    clean_captions(captions)
    tokenizer = create_tokenizer(captions)
    vocab_size = len(tokenizer.word_index) + 1
    # 1. Compute max_length from cleaned captions
    max_length = max(len(c.split()) for caps in captions.values() for c in caps)

# 2. Extract image features
    features = extract_features('../Flickr8k_Dataset')
    captions = {k: v for k, v in captions.items() if k in features}  # ⬅️ this is new


# 3. Save extracted features
    dump(features, open('../models/features.pkl', 'wb'))

# 4. Save tokenizer
    dump(tokenizer, open('../models/tokenizer.pkl', 'wb'))

# 5. Save max_length
    dump(max_length, open('../models/max_length.pkl', 'wb'))


    model = define_model(vocab_size, max_length)
    steps = len(captions) * 5 // 32

    gen = data_generator(captions, features, tokenizer, max_length, vocab_size)
    model.fit(gen, epochs=20, steps_per_epoch=steps, verbose=2)
    model.save('../models/model_cap.h5')
