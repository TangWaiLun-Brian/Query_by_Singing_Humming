import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from visualization import plot_chroma_vertical
from data import IOACAS_dataset

data_root = data_root = ".\data\IOACAS_QBH_Coprus"
epislon = 1e-5

data = IOACAS_dataset(data_root=data_root)
# print(len(data.chromagram_list))
data_list = data.chromagram_list

for i in range(len(data_list)):
    data_list[i] = data_list[i] / (np.max(data_list[i], axis=1).reshape(-1, 1) + epislon)
# plot_chroma_vertical(data_list[0].T)
# plot_chroma_vertical(data_list[1].T)

query_list = data.wav_list
# query_list = query_list
# print(len(query_list))
# for i in range(10):
#     print(query_list[i][0].shape)

def vis_extract_feature(wav_data):
    chroma1 = librosa.feature.chroma_stft(y=wav_data)
    chroma2 = librosa.feature.chroma_cens(y=wav_data)
    chroma3 = librosa.feature.chroma_cqt(y=wav_data)

    chroma1 = smoothing_downsampling(chroma1, filter_length=60, downsampling_factor=20)
    chroma2 = smoothing_downsampling(chroma2, filter_length=41, downsampling_factor=10)
    chroma3 = smoothing_downsampling(chroma3, filter_length=41, downsampling_factor=10)


    plot_chroma_vertical(chroma1, title='stft')
    plot_chroma_vertical(chroma2, title='cens')
    plot_chroma_vertical(chroma3, title='cqt')


    plt.show()

def smoothing_downsampling(chroma, filter_length=30, downsampling_factor=5, kernel_type='boxcar'):
    # smoothing
    filter_kernel = signal.get_window(kernel_type, filter_length).reshape(1, -1)
    smooth_chroma = signal.convolve(chroma, filter_kernel, mode='same') / filter_length

    # downsampling
    downsampled_chroma = smooth_chroma[:, ::downsampling_factor]
    return downsampled_chroma

extract_feature(query_list[0])
# chroma2 = librosa.feature.chroma_cens(y=query_list[0])
# smoothing_downsampling(chroma2)