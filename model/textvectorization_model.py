import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from keras.preprocessing import pad_sequences
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout, TextVectorization
from keras.models import Sequential


# x = protein sequences, y = binary lifestyle

input_shape = 18000
max_features = 18000

training_df = pd.read_csv('training_data.csv')
testing_df = pd.read_csv('testing_data.csv')
validation_df = pd.read_csv('validation_data.csv')

'''
raw_seqs = df['protein_sequence'].tolist
labels = df['Temperate (empirical)'].value
'''


#start dataframes
x_training = training_df['protein_sequence'].tolist
y_training = training_df['Binary Lifestyle']

x_testing = testing_df['protein_sequence'].tolist
y_testing = testing_df['Binary Lifestyle']

x_validation = validation_df['protein_sequence'].tolist
y_validation = validation_df['Binary Lifestyle']

vectorization_layer = TextVectorization(
    max_tokens= max_features,
    standardize="lower_and_strip_punctuation",
    output_mode="int",
    pad_to_max_tokens= True,
    output_sequence_length= 18000)

vectorization_layer.adapt(x_training)
vectorization_layer.adapt(x_testing)
vectorization_layer.adapt(x_validation)



#model architecture
model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(input_dim=21, output_dim=16,mask_zero=True))
model.add(Conv1D(filters = 128, kernel_size=15, strides=3))
model.add(MaxPooling1D(pool_size=20, strides= 5,))
model.add(Bidirectional(LSTM(units=256, activation='ReLU', dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


#
model.compile(optimizer='adam', loss='binary_crossentropy', metrics='accuracy')


train = model.fit(
    x_training, y_training,
    epochs=10,
    batch_size= 50,
    verbose=1,
    validation_data= (x_validation, y_validation)
)
