import numpy as np
from scipy import ndimage
import librosa
import libfmp.c7

def compute_constellation_maplib(Y, dist_freq=7, dist_time=7, thresh=0.01):
    return libfmp.c7.c7s1_audio_id.compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time, thresh=thresh)

def constellation_map_matchinglib(C_Q, C_D, tol_freq=1, tol_time=1):
    Delta, shift_max = libfmp.c7.c7s1_audio_id.compute_matching_function(C_D, C_Q, tol_freq=tol_freq, tol_time=tol_time)
    # print(Delta.shape, shift_max)
    # print(Delta[shift_max])
    return Delta[shift_max]

def compute_mfcc(wav_data, bin_range=[0, 100], n=2048, h=1024):
    spectrogram = librosa.feature.melspectrogram(y=wav_data, n_fft=n, hop_length=h)
    return np.abs(spectrogram[:, ::4])

def compute_spectrogram(wav_data, bin_range=[0, 100], n=2048, h=1024):
    spectrogram = librosa.stft(y=wav_data, n_fft=n, hop_length=h)
    return np.abs(spectrogram[bin_range[0]:bin_range[1], ::4])
    # return np.abs(spectrogram)

def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
    """Compute constellation map (implementation using image processing)

    Notebook: C7/C7S1_AudioIdentification.ipynb

    Args:
        Y (np.ndarray): Spectrogram (magnitude)
        dist_freq (int): Neighborhood parameter for frequency direction (kappa) (Default value = 7)
        dist_time (int): Neighborhood parameter for time direction (tau) (Default value = 7)
        thresh (float): Threshold parameter for minimal peak magnitude (Default value = 0.01)

    Returns:
        Cmap (np.ndarray): Boolean mask for peak structure (same size as Y)
    """
    result = ndimage.maximum_filter(Y, size=[2*dist_freq+1, 2*dist_time+1], mode='constant')
    Cmap = np.logical_and(Y == result, result > thresh)
    return Cmap

def constellation_map_matching(query_map, db_map):
    assert db_map.shape[1] >= query_map.shape[1]
    tol_freq = 1
    tol_time = 1

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