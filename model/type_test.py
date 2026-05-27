
import pandas as pd
import numpy as np



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


for t in x_training:
    print(type(t))
