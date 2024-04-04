import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os

import libfmp.b
import libfmp.c4
import libfmp.c7

from visualization import plot_chroma_vertical
from data import IOACAS_dataset
from preprocessing import extract_feature, smoothing_downsampling
from dynamic_time_warping import *


data_root = data_root = ".\data\IOACAS_QBH_Coprus"
epislon = 1e-5

data = IOACAS_dataset(data_root=data_root)
# print(len(data.chromagram_list))
data_list = data.chromagram_list

for i in range(len(data_list)):
    data_list[i] = data_list[i] / (np.max(data_list[i], axis=1).reshape(-1, 1) + epislon)

query_list = data.wav_list

max_score = -9999
target_ranking = []
for i, query in enumerate(query_list[0:]):
    score_list = []
    matching = []
    back_tracking = []
    target_score = 0
    target_ind = -1
    extracted_query_chroma = extract_feature(query)
    for j, db_chroma in enumerate(data_list):
        # score = Dynamic_Time_Wrapping_substring(extracted_query_chroma, smoothing_downsampling(db_chroma, filter_length=41, downsampling_factor=10))
        # var = 0
        # print(db_chroma.shape)
        downsample_db_chroma = smoothing_downsampling(db_chroma, filter_length=80, downsampling_factor=80)
        # downsample_db_chroma = db_chroma[:, ::40]
        # print(extracted_query_chroma.shape, downsample_db_chroma.shape)
        # score, var = Dynamic_Time_Wrapping_subsequence(extracted_query_chroma, downsample_db_chroma)
        score, var = Dynamic_Time_Wrapping_subsequence_cost(extracted_query_chroma, downsample_db_chroma)
        # score, var = cross_correlation(extracted_query_chroma, downsample_db_chroma)


        matching.append(j)
        score_list.append(score)
        back_tracking.append(var)
        if str(data.singing_ground_truth_ID[i]) in data.midi_file_name[j]:
            target_score = score
            target_ind = j

            # fn_wav_X = os.path.join(r'C:\Users\lun\OneDrive\Documents\CUHK\Academics\AIST\AIST3110\Project\Query_by_Singing_Humming\data\IOACAS_QBH_Coprus\IOACAS_pt1', data.singing_file_path[j])
            # ell = 21
            # d = 5
            # X, N, Fs_X, x_duration = compute_cens_from_file(fn_wav_X, ell=ell, d=d)
            # C = libfmp.c7.cost_matrix_dot(X, downsample_db_chroma)
            # D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
            # length = extracted_query_chroma.shape[1]
            #
            # fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1],
            #                                           'height_ratios': [1, 1]}, figsize=(8, 4))
            # cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
            # libfmp.b.plot_matrix(C, Fs=Fs_X, ax=[ax[0]], ylabel='Time (seconds)',
            #                      title='Cost matrix $C$ with ground truth annotations (blue rectangles)',
            #                      colorbar=False, cmap=cmap)
            # plt.show()
            # print(extracted_query_chroma.shape, db_chroma.shape)
    rank = (np.array(score_list) >= target_score).sum()
    target_ranking.append(rank)
    print(f'{i}: {rank}')

print('average ranking:', sum(target_ranking) / len(target_ranking))
score_list = np.array(score_list)
ind = np.argpartition(score_list, -15)[-15:]
best_scores = score_list[ind]
best_matches = np.array(matching)[ind]
# print(data.singing_file_path[0], data.singing_ground_truth_ID[0])

