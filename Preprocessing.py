import librosa.display
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
dataset_path = 'Audio_Dataset/'
destination_path = '/wav_train_test'

'''
LABELS :-
 0 - ANGER
 1 - BORED
 2 - DISGUST
 3 - ANXIETY
 4 - HAPPY
 5 - SAD
 6 - Neutral

GENDER :-
  1 - MALE
  0 - FEMALE
'''
labels_encoded = {'W':0, 'L':1, 'E':2, 'A':3, 'F':4, 'T':5, 'N':6}
gender_encoded = {'03':1 , '10' : 1, '11':1 , '12':1 , '15':1, '08':0, '09':0, '13':0 , '14':0, '16':0}

labels = []
file_paths = []
gender = []
for file_name in os.listdir(dataset_path):
  if file_name.endswith('.wav'):
    file_paths.append(file_name)
    labels.append(labels_encoded[file_name[5]])

df = pd.DataFrame(
    {
        'file_path':file_paths,
        'label':labels,
    }
)
print(df)
df_2_audio_vals = []
x = 1
for audio_file in df['file_path']:
  data, sampling_rate = librosa.load(f'Audio_Dataset/{audio_file}')
  data_mfcc = librosa.feature.mfcc(y = data , sr = sampling_rate, n_mfcc = 20)
  # print(data_mfcc)
  df_2_audio_vals.append(data_mfcc)

columns_length = [aud.shape[1] for aud in df_2_audio_vals]
max_col_len = max(columns_length)

## PADDING
df_2_audio_vals_padded = df_2_audio_vals.copy()
for ind, arr in enumerate(df_2_audio_vals_padded):
  arr_sh = arr.shape[1]
  zero_cols = max_col_len - arr_sh
  zero_arr = np.zeros((arr.shape[0], zero_cols))
  res_arr = np.hstack((arr, zero_arr))
  df_2_audio_vals_padded[ind] = res_arr

scaler = StandardScaler()
for ind, aud in enumerate(df_2_audio_vals):
  df_2_audio_vals[ind] = scaler.fit_transform(df_2_audio_vals[ind])
  df_2_audio_vals_padded[ind] = scaler.fit_transform(df_2_audio_vals_padded[ind])
df_2_audio_vals[230].shape

df_2 = pd.DataFrame(
    {
        'audio': df_2_audio_vals_padded,
        'labels': labels
    }
)

df_2_unpad = pd.DataFrame(
    {
        'audio':df_2_audio_vals,
        'labels':labels
    }
)
## FLATTEN THE MATRIX IN EACH CELL
df_2_flattened = df_2.copy()
df_2_flattened['audio'] = df_2_flattened['audio'].apply(lambda x : x.reshape(-1))

X = df_2_flattened['audio']
y = df_2['labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

X_train_res = X_train.to_numpy().flatten()
X_train_tens = tf.stack(X_train_res)

X_test_res = X_test.to_numpy().flatten()
X_test_tens = tf.stack(X_test_res)

X_train_tensor_res = np.reshape(X_train_tens, (X_train_tens.shape[0], 1, X_train_tens.shape[1]))
X_test_tensor_res = np.reshape(X_test_tens, (X_test_tens.shape[0], 1, X_test_tens.shape[1]))

y_train_one_hot = tf.one_hot(y_train, 7)
y_test_one_hot = tf.one_hot(y_test, 7)
def Data_Preprocessing():
    return X_train_tensor_res, y_train_one_hot,X_test_tensor_res, y_test_one_hot
