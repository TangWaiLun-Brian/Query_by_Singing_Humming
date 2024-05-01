import numpy as np
from scipy import ndimage
import librosa
import libfmp.c7

def compute_constellation_maplib(Y, dist_freq=7, dist_time=7, thresh=0.01):
    '''
    utility function for constellation map computation
    :param Y: 2D feature map
    :param dist_freq: neighborhood in frequency dimension
    :param dist_time: neighborhood in time dimension
    :param thresh: threshold for minimum peak magnitude
    :return: constellation map
    '''
    return libfmp.c7.c7s1_audio_id.compute_constellation_map(Y, dist_freq=dist_freq, dist_time=dist_time, thresh=thresh)

def constellation_map_matchinglib(C_Q, C_D, tol_freq=1, tol_time=1):
    '''
    utility function for constellation map matching
    :param C_Q: Query Constellation Map
    :param C_D: Database Constellation Map
    :param tol_freq: tolerate frequency
    :param tol_time: tolerate time
    :return: mathcing score
    '''
    Delta, shift_max = libfmp.c7.c7s1_audio_id.compute_matching_function(C_D, C_Q, tol_freq=tol_freq, tol_time=tol_time)
    return Delta[shift_max]

def compute_mel_spectrogram(wav_data, n=2048, h=1024):
    '''
    utility function for mel spectrogram
    :param wav_data: audio wave data
    :param n: window size
    :param h: hop length
    :return: mel spectrogram
    '''
    spectrogram = librosa.feature.melspectrogram(y=wav_data, n_fft=n, hop_length=h)
    return np.abs(spectrogram[:, ::4])

def compute_spectrogram(wav_data, bin_range=[0, 100], n=2048, h=1024):
    '''
    utility function for spectrogram computation
    :param wav_data: audio wave data
    :param bin_range: truncate bin range
    :param n: window size
    :param h: hop length
    :return: magnitude spectrum
    '''
    spectrogram = librosa.stft(y=wav_data, n_fft=n, hop_length=h)
    return np.abs(spectrogram[bin_range[0]:bin_range[1], ::4])

def compute_constellation_map(Y, dist_freq=7, dist_time=7, thresh=0.01):
    '''
    Compute constellation map (implementation using image processing)
    Reference: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html
    :param Y:Spectrogram
    :param dist_freq:Neighborhood in frequency dimension
    :param dist_time:Neighborhood in time dimension
    :param thresh:Minimum peak magnitude to avoid noise
    :return: consetllation map (bool mask)
    '''

    freq_length = 2 * dist_freq + 1
    time_length = 2 * dist_time + 1
    result = ndimage.maximum_filter(Y, size=[freq_length, time_length], mode='constant')
    Cmap = (Y == result) & (result > thresh)
    return Cmap

def constellation_map_matching(query_map, db_map):
    '''
    Manual constellation map matching
    Reference: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S1_AudioIdentification.html
    :param query_map: Query Constellation map
    :param db_map: Database Constellation map
    :return: matching score
    '''
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