# text_summarizer_arxiv.py
import json
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split

# TensorFlow / Keras
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Concatenate, Attention
from tensorflow.keras import backend as K

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


# Download required NLTK data
nltk.download("punkt")
nltk.download("stopwords")

# -----------------------------
# Step 1 — Parse ArXiv Metadata
# -----------------------------

print("➡️ Parsing arXiv metadata JSON snapshot...")

input_texts = []
target_texts = []

file_name = "arxiv-metadata-oai-snapshot.json"  # change if your name differs

with open(file_name, "r", encoding="utf-8") as f:
    for i, line in enumerate(tqdm(f, total=500000)):  # limit 500k lines for speed
        if i >= 10000:  # use first 100k samples
            break
        try:
            obj = json.loads(line)
            abstract = obj.get("abstract", "").strip()
            title = obj.get("title", "").strip()

            if abstract and title:
                input_texts.append(abstract)
                # add start/end tokens to the target title
                target_texts.append("sos " + title + " eos")
        except:
            continue

print(f"📊 Loaded {len(input_texts)} abstract→title pairs")

# -----------------------------
# Step 2 — Clean Text
# -----------------------------

print("➡️ Cleaning text...")

contractions = {}  # optional: load from contractions.pkl if available

stop_words = set(stopwords.words("english"))

def clean(text, src):
    text = BeautifulSoup(text, "lxml").text
    words = word_tokenize(text.lower())
    # keep alphanumeric words
    words = [w for w in words if w.isalnum() and len(w) > 2]
    # remove stopwords only on abstracts
    if src == "inputs":
        words = [w for w in words if w not in stop_words]
    return words

clean_inputs = [" ".join(clean(t, "inputs")) for t in input_texts]
clean_targets = [" ".join(clean(t, "target")) for t in target_texts]

MAX_INPUT_LEN = 300
MAX_TARGET_LEN = 30

# -----------------------------
# Step 3 — Train/Test Split
# -----------------------------
x_train, x_test, y_train, y_test = train_test_split(
    clean_inputs, clean_targets, test_size=0.2, random_state=42
)

# -----------------------------
# Step 4 — Tokenize & Pad
# -----------------------------
print("➡️ Tokenizing...")

# Input tokenizer
num_in_words = 5000
in_tokenizer = Tokenizer(num_words=num_in_words, oov_token="<OOV>")
in_tokenizer.fit_on_texts(x_train)
x_train_seq = in_tokenizer.texts_to_sequences(x_train)

# Target tokenizer
num_tr_words = 2000
tr_tokenizer = Tokenizer(num_words=num_tr_words, oov_token="<OOV>")
tr_tokenizer.fit_on_texts(y_train)
y_train_seq = tr_tokenizer.texts_to_sequences(y_train)

# Pad sequences
max_in_len = 300   # abstracts
max_tr_len = 20    # titles
en_in_data = pad_sequences(x_train_seq, maxlen=max_in_len, padding="post")
dec_data = pad_sequences(y_train_seq, maxlen=max_tr_len, padding="post")

# Decoder input / target
dec_in_data = dec_data[:, :-1]
dec_tr_data = dec_data[:, 1:].reshape(len(dec_data), max_tr_len-1, 1)

print(f"Input shape  : {en_in_data.shape}")
print(f"Decoder shape: {dec_in_data.shape}")

# -----------------------------
# Step 5 — Build Seq2Seq + Attention
# -----------------------------
print("➡️ Building model...")

latent_dim = 128

# Encoder
en_inputs = Input(shape=(max_in_len,))
en_embedding = Embedding(num_in_words+1, latent_dim)(en_inputs)

en_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
en_out1, s1_h, s1_c = en_lstm1(en_embedding)
en_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
en_out2, s2_h, s2_c = en_lstm2(en_out1)
en_lstm3 = LSTM(latent_dim, return_sequences=True, return_state=True)
en_out3, s3_h, s3_c = en_lstm3(en_out2)

en_states = [s3_h, s3_c]

# Decoder
dec_inputs = Input(shape=(None,))
dec_emb = Embedding(num_tr_words+1, latent_dim)(dec_inputs)
dec_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
dec_out, *_ = dec_lstm(dec_emb, initial_state=en_states)

# Attention
attn_layer = Attention()
attn_out = attn_layer([dec_out, en_out3])

merge = Concatenate(axis=-1)([dec_out, attn_out])
dec_dense = Dense(num_tr_words+1, activation="softmax")
dec_final = dec_dense(merge)

model = Model([en_inputs, dec_inputs], dec_final)
model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

print(model.summary())

# -----------------------------
# Step 6 — Train
# -----------------------------
print("➡️ Training model...")
model.fit(
    [en_in_data, dec_in_data],
    dec_tr_data,
    batch_size=16,
    epochs=1,
    validation_split=0.1,
)

# Save model & tokenizers
print("➡️ Saving model & tokenizers...")
model.save("arxiv_summarizer.keras")

import pickle

with open("in_tokenizer.pkl", "wb") as f:
    pickle.dump(in_tokenizer, f)

with open("tr_tokenizer.pkl", "wb") as f:
    pickle.dump(tr_tokenizer, f)

# -----------------------------
# Step 7 — Inference Model for Prediction
# -----------------------------
print("➡️ Building inference models...")

# Encoder inference
enc_model = Model(en_inputs, [en_out3, s3_h, s3_c])

# Decoder inference
dec_state_input_h = Input(shape=(latent_dim,))
dec_state_input_c = Input(shape=(latent_dim,))
dec_hidden_input = Input(shape=(max_in_len, latent_dim))

dec_emb2 = model.layers[5](dec_inputs)
dec_lstm2 = model.layers[7]
dec_out2, state_h2, state_c2 = dec_lstm2(dec_emb2, initial_state=[dec_state_input_h, dec_state_input_c])

attn_out2 = attn_layer([dec_out2, dec_hidden_input])
merge2 = Concatenate(axis=-1)([dec_out2, attn_out2])
dec_dense2 = model.layers[-1]
dec_final2 = dec_dense2(merge2)

dec_model = Model(
    [dec_inputs, dec_hidden_input, dec_state_input_h, dec_state_input_c],
    [dec_final2, state_h2, state_c2]
)

reverse_target_index = tr_tokenizer.index_word
target_index = tr_tokenizer.word_index

def decode_sequence(input_seq):
    abstract_enc, h, c = enc_model.predict(input_seq)

    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = target_index["sos"]

    decoded = ""
    stop_condition = False

    while not stop_condition:
        output, h, c = dec_model.predict([target_seq, abstract_enc, h, c])
        idx = np.argmax(output[0, -1, :])
        word = reverse_target_index.get(idx, "")
        decoded += word + " "

        if word == "eos" or len(decoded) > max_tr_len:
            stop_condition = True

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = idx

    return decoded.replace("eos", "").strip()

# -----------------------------
# Step 8 — Test Prediction
# -----------------------------
print("Example inference test")
test_abstract = x_test[0]
seq = in_tokenizer.texts_to_sequences([test_abstract])
seq = pad_sequences(seq, maxlen=max_in_len, padding="post")
print("\nAbstract:", test_abstract)
print("Predicted Title:", decode_sequence(seq))

