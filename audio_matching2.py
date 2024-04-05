import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

import libfmp.b
import libfmp.c4
import libfmp.c7

from visualization import plot_chroma_vertical, plot_spec_2
from data import IOACAS_dataset
from data2 import MIR_dataset
from preprocessing import extract_feature, smoothing_downsampling
from dynamic_time_warping import *
from audio_fingerprint import *

data_root = ".\data\MIR-QBSH-corpus"
epislon = 1e-5

data = MIR_dataset(data_root=data_root)
# print(len(data.chromagram_list))
# data_list = data.chromagram_list

# for i in range(len(data_list)):
#     data_list[i] = data_list[i] / (np.max(data_list[i], axis=1).reshape(-1, 1) + epislon)

query_list = data.wav_data_list
data_list = data.db_wav_list
print(len(query_list), len(data_list))
print(query_list[0].shape, data_list[0].shape)
max_score = -9999
target_ranking = []
for i, query in enumerate(query_list):
    score_list = []
    matching = []
    back_tracking = []
    target_score = 0
    target_ind = -1
    extracted_query_chroma = extract_feature(query)
    for j, db_wav in enumerate(data_list):
        downsample_db_chroma = data.chroma_list[j]
        # score = dtw(query, db_wav)
        # score = cross_correlation(extracted_query_chroma, downsample_db_chroma)
        # score = dtw(extracted_query_chroma, downsample_db_chroma)
        # score = Dynamic_Time_Wrapping_subsequence_cost_back(extracted_query_chroma, downsample_db_chroma)
        score = Dynamic_Time_Wrapping_subsequence_cost_back2(query, db_wav)
        score_list.append(score)

        # print((data.wav_files[i].split('\\')[-1].strip('.wav')), data.df[0][j])
        if (data.song_codes[i]) == (data.df[0][j]):
            target_score = score
            target_ind = j

    rank = (np.array(score_list) >= target_score).sum()
    target_ranking.append(rank)
    print(f'{i}: {rank}')

print('average ranking:', sum(target_ranking) / len(target_ranking))
print('hit rate:', np.sum(np.array(target_ranking) <= 10))


