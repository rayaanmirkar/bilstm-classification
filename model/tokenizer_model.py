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
    max_tokens= 25,
    standardize=None,
    output_mode="int",
    split="character")

vectorization_layer.adapt(x_training)
dim_size = len(vectorization_layer.get_vocabulary()) # maybe use vocab size method?

#model architecture
model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(input_dim=(dim_size), output_dim=16))
model.add(Conv1D(filters = 128, kernel_size=15, strides=3, activation= 'relu'))
model.add(MaxPooling1D(pool_size=20, strides= 5,))
model.add(Bidirectional(LSTM(units=128, dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

train_ds = tf.data.Dataset.from_tensor_slices((x_training, y_training)).padded_batch(8, ([None], []))
val_ds = tf.data.Dataset.from_tensor_slices((x_validation, y_validation)).padded_batch(8, ([None], []))


train = model.fit(
    train_ds,
    epochs=10,
    validation_data= val_ds,
)