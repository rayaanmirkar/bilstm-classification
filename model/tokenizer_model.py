import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.preprocessing import Tokenizer, pad_sequences
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout
from keras.models import Sequential

# x = protein sequences, y = binary lifestyle

input_shape = 18000


training_df = pd.read_csv('training_data.csv')
testing_df = pd.read_csv('testing_data.csv')
validation_df = pd.read_csv('validation_data.csv')

'''
raw_seqs = df['protein_sequence'].tolist
labels = df['Temperate (empirical)'].value
'''
# *******INITIALIZE DATAFRAMES**********
x_training_raw = training_df['protein_sequence'].tolist
y_training = training_df['Binary Lifestyle']

x_testing_raw = testing_df['protein_sequence'].tolist
y_testing = testing_df['Binary Lifestyle']

x_validation_raw = validation_df['protein_sequence'].tolist
y_validation = validation_df['Binary Lifestyle']
#****************************************

#*****INITIALIZE AND FIT TOKENIZER*******
all_texts = x_testing_raw + x_training_raw + x_validation_raw
tokenizer = Tokenizer(char_level = True, oov_token = '?')
tokenizer.fit_on_texts(all_texts)
#***************************************

#*************Sequences to integers************************
x_training_int = tokenizer.texts_to_sequences(x_training_raw)
x_testing_int = tokenizer.texts_to_sequences(x_testing_raw)
x_validation_int = tokenizer.texts_to_sequences(x_validation_raw)
#**************************************************************


#************PADDED SEQUENCES***********************************
x_padded_training = pad_sequences(x_training_int, maxlen = 18000, padding = "post", truncating = 'post')
x_padded_testing = pad_sequences(x_testing_int, maxlen = 18000, padding = "post", truncating = 'post')
x_padded_validation = pad_sequences(x_validation_int, maxlen = 18000, padding = "post", truncating = 'post')
#*******************************************************************************************

# the padded training ones are final 


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


train = model.fit(
    x_padded_training, y_training,
    epochs=10,
    batch_size= 50,
    verbose=1,
    validation_data= (x_padded_validation, y_validation)
)

