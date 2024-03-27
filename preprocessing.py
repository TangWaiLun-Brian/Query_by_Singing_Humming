import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from visualization import plot_chroma_vertical
from data import IOACAS_dataset



def extract_feature(wav_data, vis=False):
    chroma1 = librosa.feature.chroma_stft(y=wav_data)
    chroma2 = librosa.feature.chroma_cens(y=wav_data)
    chroma3 = librosa.feature.chroma_cqt(y=wav_data)

    chroma1 = smoothing_downsampling(chroma1, filter_length=60, downsampling_factor=20)
    chroma2 = smoothing_downsampling(chroma2, filter_length=10, downsampling_factor=20)
    chroma3 = smoothing_downsampling(chroma3, filter_length=41, downsampling_factor=10)

    if vis:
        plot_chroma_vertical(chroma1, title='stft')
        plot_chroma_vertical(chroma2, title='cens')
        plot_chroma_vertical(chroma3, title='cqt')


        plt.show()
    return chroma2

def smoothing_downsampling(chroma, filter_length=30, downsampling_factor=5, kernel_type='boxcar'):
    # smoothing
    filter_kernel = signal.get_window(kernel_type, filter_length).reshape(1, -1)
    smooth_chroma = signal.convolve(chroma, filter_kernel, mode='same') / filter_length

    # downsampling
    downsampled_chroma = smooth_chroma[:, ::downsampling_factor]
    return downsampled_chroma

# print(data.singing_file_path[0], data.singing_ground_truth_ID[0])
# extract_feature(query_list[0])

# chroma2 = librosa.feature.chroma_cens(y=query_list[0])
# smoothing_downsampling(chroma2)