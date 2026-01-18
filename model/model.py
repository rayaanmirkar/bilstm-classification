import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.preprocessing import Tokenizer, pad_sequences
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout
from keras.models import Sequential

input_shape = 18000

df = pd.read_csv('final_data_training.csv')

raw_seqs = df['protein_sequence'].tolist
labels = df['Temperate (empirical)'].values

tokenizer = Tokenizer(char_level = True, oov_token = '?')
tokenizer.fit_on_texts(raw_seqs)

int_seqs = tokenizer.texts_to_sequences(raw_seqs)
padded_data = pad_sequences(int_seqs, maxlen = 18000, padding = "post", truncating = 'post')


#model architecture
model = Sequential()

model.add(Embedding(input_dim=21, output_dim=16,mask_zero=True))
model.add(Conv1D(filters = 128, kernel_size=15, strides=3))
model.add(MaxPooling1D(pool_size=20, strides= 5,))
model.add(Bidirectional(LSTM(units=256, activation='ReLU', dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


#
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')




