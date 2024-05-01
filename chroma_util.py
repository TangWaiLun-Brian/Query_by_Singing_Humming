import numpy as np
from scipy import ndimage
import librosa
import libfmp.c7

def chroma_constellation_map_matching(query_map, db_map, tol_freq=1, tol_time=1):
    '''
    perform chroma consetllation map matching
    Reference: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html
    :param query_map: Query map
    :param db_map: Database map
    :param tol_freq: tolerate frequency
    :param tol_time: tolerate time
    :return: matching score
    '''
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
    '''
    chroma matching with binary mask
    :param C_X: Query map
    :param C_Y: Database map
    :param tol_freq: tolerate frequency
    :param tol_time: tolerate time
    :return: mathcing score
    '''
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
    '''
    utility function for dynamic time warping on chroma features
    :param C_X: Query chroma
    :param C_Y: Database chroma
    :return: matching score
    '''
    Delta, C, D = libfmp.c7.c7s2_audio_matching.compute_matching_function_dtw(C_X, C_Y, stepsize=2)
    return -D[-1, -1] / C_Y.shape[1]

def chroma_matching_dtw_subsequence_shift(C_X, C_Y):
    '''
    utility function for subsequence dtw on chroma features
    :param C_X: query chroma
    :param C_Y: database chroma
    :return: matching score
    '''

    score = 999999
    for i in range(12):
        C_X = libfmp.c3.c3s1_transposition_tuning.cyclic_shift(C_X, shift=1)
        C = libfmp.c7.c7s2_audio_matching.cost_matrix_dot(C_X, C_Y)
         # Delta, C, D = libfmp.c7.c7s2_audio_matching.compute_matching_function_dtw(C_X, C_Y, stepsize=2)
        D = libfmp.c7.c7s2_audio_matching.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
        score = min(score, D[-1, -1])
    return -score / C_Y.shape[1]
