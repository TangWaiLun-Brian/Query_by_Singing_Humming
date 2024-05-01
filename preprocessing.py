import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from visualization import plot_chroma_vertical
from data import IOACAS_dataset

def smoothing_downsampling2(feature, filter_length=30, downsampling_factor=5, kernel_type='boxcar'):
    '''
    utility function for smoothing and downsampling on one dimensional features
    Reference: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C3/C3S1_FeatureSmoothing.html
    :param feature: 1d feature
    :param filter_length: smoothing window size
    :param downsampling_factor: factor for downsampling
    :param kernel_type: the choice of kernel
    :return:
    '''
    # smoothing
    filter_kernel = signal.get_window(kernel_type, filter_length).reshape(-1)
    smooth_feature = signal.convolve(feature, filter_kernel, mode='same') / filter_length

    # downsampling
    downsampled_feature = smooth_feature[::downsampling_factor]
    return downsampled_feature

def extract_feature(wav_data, vis=False):
    '''
    extracting chroma features
    :param wav_data: wave data
    :param vis: visualize chromagrams
    :return: chroma
    '''

    wav_data = smoothing_downsampling2(wav_data, filter_length=1000, downsampling_factor=4)
    chroma1 = librosa.feature.chroma_stft(y=wav_data, n_fft=2048, hop_length=512)[:, :]
    chroma2 = librosa.feature.chroma_cens(y=wav_data, hop_length=512)[:, :]
    # chroma3 = librosa.feature.chroma_cqt(y=wav_data, hop_length=512)[:, :]

    # chroma1 = smoothing_downsampling(chroma1, filter_length=3, downsampling_factor=3)
    # chroma2 = smoothing_downsampling(chroma2, filter_length=3, downsampling_factor=3)
    # chroma3 = smoothing_downsampling(chroma3, filter_length=10, downsampling_factor=10)

    if vis:
        plot_chroma_vertical(chroma1, title='stft')
        plot_chroma_vertical(chroma2, title='cens')
        # plot_chroma_vertical(chroma3, title='cqt')


        plt.show()
    return (chroma1 + chroma2) / 2

def smoothing_downsampling(chroma, filter_length=30, downsampling_factor=5, kernel_type='boxcar'):
    '''
    utility function for smoothing and downsampling for 2 dimensional features
    :param chroma: chroma
    :param filter_length: smoothing window size
    :param downsampling_factor: downsampling rate
    :param kernel_type: choice of kernel
    :return: downsampled and smoothed chroma
    '''
    # smoothing
    filter_kernel = signal.get_window(kernel_type, filter_length).reshape(1, -1)
    smooth_chroma = signal.convolve(chroma, filter_kernel, mode='same') / filter_length

    # downsampling
    downsampled_chroma = smooth_chroma[:, ::downsampling_factor]
    return downsampled_chroma
