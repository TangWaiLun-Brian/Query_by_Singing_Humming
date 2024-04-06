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
from chroma_fingerprint import *

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
    # print(query.shape, extracted_query_chroma.shape)
    for j, db_wav in enumerate(data_list):
        downsample_db_chroma = data.chroma_list[j][:, ::4]

        # spectrogram fingerprint best param
        # dis_freq = 7
        # dis_time = 1
        # tol_freq = 3
        # tol_time = 1

        # chroma fingerprint param
        # dis_freq = 3
        # dis_time = 1
        # tol_freq = 1
        # tol_time = 1

        # spectro fingerprint
        # q = smoothing_downsampling2(query, filter_length=1, downsampling_factor=4)
        # db = smoothing_downsampling2(db_wav, filter_length=1, downsampling_factor=4)
        X = compute_spectrogram(query)
        Y = compute_spectrogram(db_wav)
        # C_X = compute_constellation_maplib(X, dist_freq=dis_freq, dist_time=dis_time)
        # C_Y = compute_constellation_maplib(Y, dist_freq=dis_freq, dist_time=dis_time)
        # score = constellation_map_matchinglib(C_X, C_Y, tol_freq=tol_freq, tol_time=tol_time)

        # spectro dtw
        score = Dynamic_Time_Wrapping_subsequence_chroma(X, Y)

        # chroma fingerprint, binary mask
        # C_X = compute_chroma_constellation_map(extracted_query_chroma, dist_freq=dis_freq, dist_time=dis_time)
        # C_Y = compute_chroma_constellation_map(downsample_db_chroma, dist_freq=dis_freq, dist_time=dis_time)
        # score = chroma_constellation_map_matching(C_X, C_Y, tol_freq=tol_freq, tol_time=tol_time)
        # score = chroma_matching_binary(C_X, C_Y, tol_freq=tol_freq, tol_time=tol_time)

        # chroma dtw, cross correlation, dot cost, norm cost
        # score = chroma_matching_dtw_subsequence_shift(extracted_query_chroma, downsample_db_chroma)
        # score = cross_correlation_chroma(extracted_query_chroma, downsample_db_chroma)
        # score = Dynamic_Time_Wrapping_subsequence_dot(extracted_query_chroma, downsample_db_chroma)
        # score = Dynamic_Time_Wrapping_subsequence_norm(extracted_query_chroma, downsample_db_chroma)

        # wav dtw
        # score = Dynamic_Time_Wrapping_subsequence_wav(query, db_wav)
        # score = slicing_dtw(query, db_wav)


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


