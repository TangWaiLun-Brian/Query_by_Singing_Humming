import numpy as np
from scipy import ndimage
import librosa
import libfmp.c7

def compute_chroma_constellation_map(Y, dist_freq=1, dist_time=1, thresh=0.01):
    """Compute constellation map (implementation using image processing)

    Notebook: C7/C7S1_AudioIdentification.ipynb

    Args:
        Y (np.ndarray): Chroma (magnitude)
        dist_freq (int): Neighborhood parameter for frequency direction (kappa) (Default value = 7)
        dist_time (int): Neighborhood parameter for time direction (tau) (Default value = 7)
        thresh (float): Threshold parameter for minimal peak magnitude (Default value = 0.01)

    Returns:
        Cmap (np.ndarray): Boolean mask for peak structure (same size as Y)
    """
    result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
    Cmap = np.logical_and(Y == result, result > thresh)
    return Cmap

def chroma_constellation_map_matching(query_map, db_map, tol_freq=1, tol_time=1):
    assert db_map.shape[1] >= query_map.shape[1]


    n = query_map.shape[1]
    m = db_map.shape[1] - n
    best_score = 0
    for i in range(m+1):
        db_sub_map = db_map[:, i:i+n]
        C_est_max = ndimage.maximum_filter(query_map, size=(2 * tol_freq + 1, 2 * tol_time + 1),
                                           mode='constant')
        tp = np.sum(np.logical_and(C_est_max, db_sub_map))
        best_score = max(best_score, tp)
    return best_score



def chroma_matching_binary(C_X, C_Y, tol_freq=0, tol_time=0):
    C_X1 = np.zeros_like(C_X)
    C_Y1 = np.zeros_like(C_Y)

    C_X1[:, np.argmax(C_X, axis=0)] = 1
    C_Y1[:, np.argmax(C_Y, axis=0)] = 1

    n = C_X.shape[1]
    m = C_Y.shape[1] - n
    best_score = -999999
    for i in range(m + 1):
        C_Y_sub = C_Y1[:, i:i + n]
        TP, FN, FP, C_AND = libfmp.c7.c7s1_audio_id.match_binary_matrices_tol(C_Y_sub, C_X1, tol_freq=tol_freq, tol_time=tol_time)
        score = TP - FP
        best_score = max(best_score, score)
    return best_score

def chroma_matching_dtw(C_X, C_Y):
    Delta, C, D = libfmp.c7.c7s2_audio_matching.compute_matching_function_dtw(C_X, C_Y, stepsize=2)
    return -D[-1, -1] / C_Y.shape[1]

def chroma_matching_dtw_subsequence_shift(C_X, C_Y):
    # C_X, fs = libfmp.c7.c7s2_audio_matching.compute_cens_from_chromagram(C_X)
    # C_Y, fs = libfmp.c7.c7s2_audio_matching.compute_cens_from_chromagram(C_Y)
    score = 999999
    for i in range(12):
        C_X = libfmp.c3.c3s1_transposition_tuning.cyclic_shift(C_X, shift=1)
        C = libfmp.c7.c7s2_audio_matching.cost_matrix_dot(C_X, C_Y)
         # Delta, C, D = libfmp.c7.c7s2_audio_matching.compute_matching_function_dtw(C_X, C_Y, stepsize=2)
        D = libfmp.c7.c7s2_audio_matching.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
        score = min(score, D[-1, -1])
    return -score / C_Y.shape[1]
