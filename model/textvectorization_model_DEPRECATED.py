import tensorflow as tf
import keras
import pandas as pd
import numpy as np
#from keras.preprocessing import pad_sequences
from keras.layers import Bidirectional
from keras.layers import Embedding, Dense, LSTM, Input, Conv1D, MaxPooling1D, Dropout, TextVectorization
from keras.models import Sequential


# x = protein sequences, y = binary lifestyle
# input_shape = 18000
max_features = 18000
training_df = pd.read_csv(r'C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\training_data.csv')
testing_df = pd.read_csv(r'C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\testing_data.csv')
validation_df = pd.read_csv(r"C:\Users\\raypi\\coding\\SoftwareProjects\\phager\\building_data\\validation_data.csv")






print(validation_df['Binary Lifestyle'].value_counts(dropna=False))
print(validation_df['Binary Lifestyle'].dtype)
print(validation_df[validation_df['Binary Lifestyle'].isna()])





#start dataframes
# training_df['protein_sentence']
x_training = (training_df['protein_sentence']).astype(str).tolist()
y_training = (training_df['Binary Lifestyle']).astype(int).tolist()

x_testing = (testing_df['protein_sentence']).astype(str).tolist()
y_testing = testing_df['Binary Lifestyle'].astype(int).tolist()

x_validation = validation_df['protein_sentence'].astype(str).tolist()
y_validation = validation_df['Binary Lifestyle'].astype(int).tolist()











vectorization_layer = TextVectorization(
    max_tokens= 25,
    standardize=None,
    output_mode="int",
    pad_to_max_tokens= True,
    output_sequence_length= 5000, 
    split="character")

vectorization_layer.adapt(x_training)

#print(len(vectorization_layer.get_vocabulary()))


#model architecture
model = Sequential()
model.add(vectorization_layer)
model.add(Embedding(input_dim=26, output_dim=16))
model.add(Conv1D(filters = 128, kernel_size=15, strides=3, activation= 'relu'))
model.add(MaxPooling1D(pool_size=20, strides= 5,))
model.add(Bidirectional(LSTM(units=128, dropout=0.2)))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))


#
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


train = model.fit(
    x_training, y_training,
    epochs=10,
    batch_size= 8,
    verbose=1,
    validation_data= (x_validation, y_validation),
)
