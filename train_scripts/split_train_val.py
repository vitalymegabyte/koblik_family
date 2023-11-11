import os

import numpy as np
from sklearn.metrics import confusion_matrix

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm

from glob import glob

import ffmpeg

import warnings
warnings.filterwarnings('ignore')


classes = {'простой': 0,
        'вынужденная': 1,
        'сварка': 2}


df_nine_left = pd.read_csv('nine_hour_left.csv')
df_nine_right = pd.read_csv('nine_hour_right.csv')

df_five_left = pd.read_csv('five_hour_left.csv')
df_five_right = pd.read_csv('five_hour_right.csv')



ann_train = []
ann_test = []


for _, r in tqdm(df_nine_left.iterrows()):
    file_result = f'train_5s/{r["fname"]}'
    line = file_result + ' ' + str(classes[r['label']]) + '\n'
    if r['time'] > 7200:
        ann_train.append(line)
    else:
        ann_test.append(line)

for _, r in tqdm(df_nine_right.iterrows()):
    file_result = f'train_5s/{r["fname"]}'
    line = file_result + ' ' + str(classes[r['label']]) + '\n'
    if r['time'] > 7200:
        ann_train.append(line)
    else:
        ann_test.append(line)

for _, r in tqdm(df_five_left.iterrows()):
    file_result = f'train_5s/{r["fname"]}'
    line = file_result + ' ' + str(classes[r['label']]) + '\n'
    if r['time'] < 13845:
        ann_train.append(line)
    else:
        ann_test.append(line)

for _, r in tqdm(df_five_right.iterrows()):
    file_result = f'train_5s/{r["fname"]}'
    line = file_result + ' ' + str(classes[r['label']]) + '\n'
    if r['time'] < 13845:
        ann_train.append(line)
    else:
        ann_test.append(line)


with open('ann_train.txt', 'w') as train_file, open('ann_test.txt', 'w') as test_file:
    train_file.writelines(ann_train)
    test_file.writelines(ann_test)