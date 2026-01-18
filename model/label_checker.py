import pandas as pd
import numpy as np
import csv


df = pd.read_csv('final_data_training.csv')
labels = df['Temperate (empirical)'].values

print(labels)