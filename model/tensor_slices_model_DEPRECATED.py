import tensorflow as tf
import keras
import pandas as pd
import numpy as np
#from keras.preprocessing import pad_sequences
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout, TextVectorization
from keras.models import Sequential


# x = protein sequences, y = binary lifestyle

input_shape = 18000
max_features = 18000

training_df = pd.read_csv(r'C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\training_data.csv')
testing_df = pd.read_csv(r'C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\testing_data.csv')
validation_df = pd.read_csv(r"C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\validation_data.csv")

'''
raw_seqs = df['protein_sequence'].tolist
labels = df['Temperate (empirical)'].value
'''


#start dataframes

x_training = training_df['protein_sentence'].astype(str).tolist()
y_training = training_df['Binary Lifestyle'].tolist()

x_testing = testing_df['protein_sentence'].astype(str).tolist()
y_testing = testing_df['Binary Lifestyle'].tolist()

x_validation = validation_df['protein_sentence'].astype(str).tolist()
y_validation = validation_df['Binary Lifestyle'].tolist()

x_training_dataset = tf.data.Dataset.from_tensor_slices(x_training).batch(32)
x_testing_dataset = tf.data.Dataset.from_tensor_slices(x_testing).batch(32)
x_validation_dataset = tf.data.Dataset.from_tensor_slices(x_validation).batch(32)


vectorization_layer = TextVectorization(
    max_tokens= 25,
    standardize="lower_and_strip_punctuation",
    output_mode="int",
    pad_to_max_tokens= True,
    output_sequence_length= 18000, 
    split="character")


vectorization_layer.adapt(x_training_dataset)
vectorization_layer.adapt(x_testing_dataset)
vectorization_layer.adapt(x_validation_dataset)



#model architecture
model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(input_dim=21, output_dim=16,mask_zero=True))
model.add(Conv1D(filters = 128, kernel_size=15, strides=3))
model.add(MaxPooling1D(pool_size=20, strides= 5,))
model.add(Bidirectional(LSTM(units=256, activation='relu', dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


#
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train = model.fit(
    x_training, y_training,
    epochs=10,
    batch_size= 50,
    verbose=1,
    validation_data= (x_validation, y_validation)
)
