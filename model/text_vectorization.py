import tensorflow as tf
import keras
import pandas as pd
import numpy as np
#from keras.preprocessing import pad_sequences
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout, TextVectorization
from keras.models import Sequential

max_features = 18000

training_df = pd.read_csv(r'C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\training_data.csv')
testing_df = pd.read_csv(r'C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\testing_data.csv')
validation_df = pd.read_csv(r"C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\validation_data.csv")

training_df = training_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])
testing_df = testing_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])
validation_df = validation_df.dropna(subset=['protein_sentence', 'Binary Lifestyle'])


x_training = (training_df['protein_sentence']).astype(str).tolist()
y_training = (training_df['Binary Lifestyle']).to_numpy(dtype=np.float32)

x_testing = (testing_df['protein_sentence']).astype(str).tolist()
y_testing = testing_df['Binary Lifestyle'].to_numpy(dtype=np.float32)

x_validation = validation_df['protein_sentence'].astype(str).tolist()
y_validation = validation_df['Binary Lifestyle'].to_numpy(dtype=np.float32)


vectorization_layer = TextVectorization(
    max_tokens= 30,
    standardize=None,
    output_mode="int",
    output_sequence_length=50000,
    split="character")

vectorization_layer.adapt(x_training)
dim_size = vectorization_layer.vocabulary_size() # maybe use vocab size method?

#model architecture
model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(input_dim=(dim_size+1), output_dim=16))
model.add(Conv1D(filters = 128, kernel_size=15, strides=3, activation= 'relu'))
model.add(Bidirectional(LSTM(units=128, dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_ds = tf.data.Dataset.from_tensor_slices((x_training, y_training)).batch(8)
val_ds = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).batch(8)


train = model.fit(
    train_ds,
    epochs=10,
    validation_data= val_ds,
)