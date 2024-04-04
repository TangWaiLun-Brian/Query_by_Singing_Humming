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

def cross_correlation(query_chroma, db_chroma):
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
    vertical_spacing = vertical[1:] - vertical[:-1]
    penalty = np.abs(vertical_spacing[1:] - vertical_spacing[:-1]) ** 2 * 2
    return -(Match_Matrix[n, m] + np.sum(penalty))

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
