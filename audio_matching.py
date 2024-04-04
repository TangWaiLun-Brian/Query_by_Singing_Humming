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

def compute_cens_from_file(fn_wav, Fs=22050, N=4410, H=2205, ell=21, d=5):
    """Compute CENS features from file

    Notebook: C7/C7S2_AudioMatching.ipynb

    Args:
        fn_wav (str): Filename of wav file
        Fs (scalar): Feature rate of wav file (Default value = 22050)
        N (int): Window size for STFT (Default value = 4410)
        H (int): Hop size for STFT (Default value = 2205)
        ell (int): Smoothing length (Default value = 21)
        d (int): Downsampling factor (Default value = 5)

    Returns:
        X_CENS (np.ndarray): CENS features
        L (int): Length of CENS feature sequence
        Fs_CENS (scalar): Feature rate of CENS features
        x_duration (float): Duration (seconds) of wav file
    """
    x, Fs = librosa.load(fn_wav, sr=Fs)
    x_duration = x.shape[0] / Fs
    X_chroma = librosa.feature.chroma_stft(y=x, sr=Fs, tuning=0, norm=None, hop_length=H, n_fft=N)
    X_CENS, Fs_CENS = libfmp.c7.compute_cens_from_chromagram(X_chroma, Fs=Fs/H, ell=ell, d=d)
    L = X_CENS.shape[1]
    return X_CENS, L, Fs_CENS, x_duration

def Dynamic_Time_Wrapping_subsequence_cost_back(query_chroma, db_chroma):
    C = libfmp.c7.cost_matrix_dot(query_chroma, db_chroma)
    D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
    length = query_chroma.shape[1]

    # fig, ax = plt.subplots(2, 1, gridspec_kw={'width_ratios': [1],
    #                                           'height_ratios': [1, 1]}, figsize=(8, 4))
    # cmap = libfmp.b.compressed_gray_cmap(alpha=-10, reverse=True)
    # libfmp.b.plot_matrix(C, ax=[ax[0]], ylabel='Time (seconds)',
    #                      title='Cost matrix $C$ with ground truth annotations (blue rectangles)',
    #                      colorbar=False, cmap=cmap)
    # plt.show()
    return -D[-1, -1] / length, (np.ones((length,)), np.ones((length,)))

def naive_exhaustion(query_chroma, db_chroma):
    n = db_chroma.shape[1] - query_chroma.shape[1] + 1
    length = query_chroma.shape[1]
    max_score = -9999
    for i in range(n):
        score = db_chroma[:, i:i+length].reshape(-1) @ query_chroma[:, :].reshape(-1)
        max_score = max(max_score, score)

    return max_score, (np.ones((length,)), np.ones((length,)))

def Dynamic_Time_Wrapping_subsequence_cost(query_chroma, db_chroma):
    n = db_chroma.shape[1]
    m = query_chroma.shape[1]
    Match_Matrix = np.zeros((n+1, m+1))
    Back_Tracking_Matrix = np.zeros((n+1, m+1))
    Track_dir = [-1, -m-2, -m-1]
    # for i in range(1, m+1):
    #     Match_Matrix[1, i] = Match_Matrix[1, i-1] + np.linalg.norm(query_chroma[:, i-1] - db_chroma[:, 0])
    # for j in range(1, n+1):
    #     Match_Matrix[j, 1] = Match_Matrix[j-1, 1] + np.linalg.norm(query_chroma[:, 0] - db_chroma[:, j-1])
    # print(np.max(db_chroma), np.min(db_chroma), np.max(query_chroma), np.min(query_chroma))
    for i in range(1, n+1):
        for j in range(1, m+1):
            # cost = np.linalg.norm(db_chroma[:, i - 1] - query_chroma[:, j - 1]) - np.sqrt(12)
            cost = -(db_chroma[:, i - 1] @ query_chroma[:, j - 1])
            dependency = np.array([Match_Matrix[i, j-1], Match_Matrix[i-1, j-1] + cost, Match_Matrix[i-1, j]])
            index = np.argmin(dependency)
            Match_Matrix[i, j] = dependency[index]
            Back_Tracking_Matrix[i, j] = Track_dir[index]


    index = (n+1) * (m+1) - 1
    i = int(index) // (m+1)
    j = int(index) % (m+1)
    subsequence = []
    while Back_Tracking_Matrix[i, j] != 0:
        # print(i, j, Back_Tracking_Matrix[i, j])
        subsequence.append(index)
        index += Back_Tracking_Matrix[i, j]
        i = int(index) // (m + 1)
        j = int(index) % (m + 1)

    subsequence = np.array(subsequence)
    horizontal = subsequence % (m+1)
    vertical = subsequence // (m+1)

    return -Match_Matrix[n, m], (horizontal, vertical)

def Dynamic_Time_Wrapping_subsequence_cost1(query_chroma, db_chroma):
    n = db_chroma.shape[1]
    m = query_chroma.shape[1]
    Match_Matrix = np.zeros((n+1, m+1))
    Back_Tracking_Matrix = np.zeros((n+1, m+1))
    Track_dir = [-1, -m-2, -m-1]
    for i in range(1, m+1):
        Match_Matrix[1, i] = Match_Matrix[1, i-1] + np.linalg.norm(query_chroma[:, i-1] - db_chroma[:, 0])
    # for j in range(1, n+1):
    #     Match_Matrix[j, 1] = Match_Matrix[j-1, 1] + np.linalg.norm(query_chroma[:, 0] - db_chroma[:, j-1])

    for i in range(2, n+1):
        for j in range(2, m+0):
            dependency = np.array([Match_Matrix[i, j-1], Match_Matrix[i-1, j-1], Match_Matrix[i-1, j]])
            index = np.argmin(dependency)
            Match_Matrix[i, j] = dependency[index] + np.linalg.norm(db_chroma[:, i - 1] - query_chroma[:, j - 1])
            Back_Tracking_Matrix[i, j] = Track_dir[index]

        dependency = np.array([Match_Matrix[i, m - 1], Match_Matrix[i - 1, m - 1], Match_Matrix[i - 1, m]])
        index = np.argmin(dependency)
        Match_Matrix[i, m] = dependency[index]
        if index != 2:
            Match_Matrix[i, m] += np.linalg.norm(db_chroma[:, i - 1] - query_chroma[:, m - 1])
        Back_Tracking_Matrix[i, m] = Track_dir[index]

    index = (n+1) * (m+1) - 1
    i = int(index) // (m+1)
    j = int(index) % (m+1)
    subsequence = []
    while Back_Tracking_Matrix[i, j] != 0:
        # print(i, j, Back_Tracking_Matrix[i, j])
        subsequence.append(index)
        index += Back_Tracking_Matrix[i, j]
        i = int(index) // (m + 1)
        j = int(index) % (m + 1)

    subsequence = np.array(subsequence)
    horizontal = subsequence % (m+1)
    vertical = subsequence // (m+1)
    spacing = int(np.mean(vertical[:-1] - vertical[1:]))
    # for k in horizontal:
    #     k = int(k)
    #     back = min(db_chroma.shape[1]-1, k * spacing + i-1)
    #     if k * spacing + i-1 >= db_chroma.shape[1]:
    #         break
    #     Match_Matrix[n, m] += (query_chroma[:, k-1] @ db_chroma[:, back])
    return -Match_Matrix[n, m], (horizontal, vertical)

def Dynamic_Time_Wrapping_subsequence(query_chroma, db_chroma):
    n = db_chroma.shape[1]
    m = query_chroma.shape[1]
    Match_Matrix = np.zeros((n+1, m+1))
    Back_Tracking_Matrix = np.zeros((n+1, m+1))
    Track_dir = [-1, -m-2, -m-1]
    for i in range(1, n+1):
        for j in range(1, m+1):
            dependency = np.array([Match_Matrix[i, j-1], Match_Matrix[i-1, j-1] + (db_chroma[:, i - 1] @ query_chroma[:, j - 1]), Match_Matrix[i-1, j]])
            index = np.argmax(dependency)
            Match_Matrix[i, j] = dependency[index]
            Back_Tracking_Matrix[i, j] = Track_dir[index]


    index = (n+1) * (m+1) - 1
    i = int(index) // (m+1)
    j = int(index) % (m+1)
    subsequence = []
    while Back_Tracking_Matrix[i, j] != 0:
        # print(i, j, Back_Tracking_Matrix[i, j])
        if Back_Tracking_Matrix[i, j] == Track_dir[1]:
            subsequence.append(index)
        index += Back_Tracking_Matrix[i, j]
        i = int(index) // (m + 1)
        j = int(index) % (m + 1)

    subsequence = np.array(subsequence)
    horizontal = subsequence % (m+1)
    vertical = subsequence // (m+1)
    spacing = int(np.mean(vertical[:-1] - vertical[1:]))
    # for k in horizontal:
    #     k = int(k)
    #     back = min(db_chroma.shape[1]-1, k * spacing + i-1)
    #     if k * spacing + i-1 >= db_chroma.shape[1]:
    #         break
    #     Match_Matrix[n, m] += (query_chroma[:, k-1] @ db_chroma[:, back])
    return Match_Matrix[n, m], (horizontal, vertical)

def Dynamic_Time_Wrapping_substring(query_chroma, db_chroma):
    n = db_chroma.shape[1]
    m = query_chroma.shape[1]
    Match_Matrix = np.zeros((n+1, m+1))
    Back_Tracking_Matrix = np.zeros((n+1, m+1))
    Track_dir = [-1, -n-2, -n-1]
    for i in range(1, n+1):
        for j in range(1, m):
            dependency = np.array([Match_Matrix[i, j-1], Match_Matrix[i-1, j-1], Match_Matrix[i-1, j]])
            index = np.argmax(dependency)
            Match_Matrix[i, j] = dependency[index] + (db_chroma[:, i - 1] @ query_chroma[:, j - 1])
            Back_Tracking_Matrix[i, j] = Track_dir[index]

        dependency = np.array([Match_Matrix[i, m-1], Match_Matrix[i - 1, m-1], Match_Matrix[i - 1, m]])
        index = np.argmax(dependency)
        Match_Matrix[i, m] = dependency[index]
        if index == 1:
            Match_Matrix[i, m] += (db_chroma[:, i-1] @ query_chroma[:, m-1])
        Back_Tracking_Matrix[i, m] = Track_dir[index]

    return Match_Matrix[n, m]

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
        # score, var = naive_exhaustion(extracted_query_chroma, downsample_db_chroma)


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
    print(f'{i}: {rank}')
    if i >= 10:
        break
score_list = np.array(score_list)
ind = np.argpartition(score_list, -15)[-15:]
best_scores = score_list[ind]
best_matches = np.array(matching)[ind]
print(data.singing_file_path[0], data.singing_ground_truth_ID[0])
for i, (best_match, best_score) in enumerate(zip(best_matches, best_scores)):
    print(data.midi_file_name[best_match], best_score)
    # print(np.mean(back_tracking[ind[i]][1]), np.var(back_tracking[ind[i]][1]) ** (1/2))
    spacing = back_tracking[ind[i]][1][:-1] - back_tracking[ind[i]][1][1:]
    # print(np.mean(spacing))

print('target score:', target_score, data.midi_file_name[target_ind])
spacing = back_tracking[target_ind][1][:-1] - back_tracking[target_ind][1][1:]
# print(np.mean(spacing))
# print(np.mean(back_tracking[target_ind][1]), np.var(back_tracking[target_ind][1]) ** (1/2))
