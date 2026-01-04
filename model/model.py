import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.preprocessing import Tokenizer, pad_sequences
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input
from keras.models import Sequential

#GET VALUES FROM CSV FILE
df = pd.read_csv('final_data_training.csv')
raw_seqs = df['protein_sequence'].tolist
labels = df['Temperate (empirical)'].values

#CONVERT PROTEONOMES TO INTEGER FORMAT VIA TOKENIZER (REPLACE w/ TXTVECTORIZATION??)
tokenizer = Tokenizer(char_level = True, oov_token = '?')
tokenizer.fit_on_texts(raw_seqs)
int_seqs = tokenizer.texts_to_sequences(raw_seqs)
padded_data = pad_sequences(int_seqs, maxlen = 18000, padding = "post", truncating = 'post')

#MODEL ARCHITECTURE

model = Sequential([

    Input.shape





])




