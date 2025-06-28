import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

with open("model/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAXLEN = 100

def preprocess_text(text: str):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAXLEN)
    return padded