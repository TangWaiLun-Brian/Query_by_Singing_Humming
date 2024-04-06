import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from visualization import plot_chroma_vertical
from data import IOACAS_dataset

def smoothing_downsampling2(feature, filter_length=30, downsampling_factor=5, kernel_type='boxcar'):
  # smoothing
  filter_kernel = signal.get_window(kernel_type, filter_length).reshape(-1)
  smooth_feature = signal.convolve(feature, filter_kernel, mode='same') / filter_length

  # downsampling
  downsampled_feature = smooth_feature[::downsampling_factor]
  return downsampled_feature

def extract_feature(wav_data, vis=False):
    # direct hardcode
    # chroma1 = librosa.feature.chroma_stft(y=wav_data, n_fft=2048, hop_length=512)[:, 200:220]

    # hardcode local consecutive indices
    # index = [[i * 50 + j for j in range(5)] for i in range(1, 5)]
    # index = np.array(index).reshape(-1)
    # chroma1 = librosa.feature.chroma_stft(y=wav_data, n_fft=2048, hop_length=512)[:, index]

    # use energy novelty function
    # sr = 22050
    # rmse = librosa.feature.rms(y=wav_data, frame_length=2048, hop_length=512)[0].reshape(-1)
    # rmse_diff = np.zeros_like(rmse)
    # rmse_diff[1:] = rmse[1:] - rmse[:-1]
    # energy_novelty = (rmse_diff + np.abs(rmse_diff)) / 2
    # energy_novelty[energy_novelty < np.max(energy_novelty) / 3] = 0
    # # frames = np.arange(len(rmse))
    # # t = librosa.frames_to_time(frames, sr=sr)
    # # plt.figure(figsize=(15, 6))
    # # plt.plot(t, rmse, 'b--', t, rmse_diff, 'g--^', t, energy_novelty, 'r-')
    # # plt.xlim(0, t.max())
    # # plt.xlabel('Time (sec)')
    # # plt.legend(('RMSE', 'delta RMSE', 'energy novelty'))
    # # plt.show()
    # index = np.array([0, 1, 2, 3, 4])
    # print(np.where(energy_novelty > 0)[0].shape)
    # local_maximum = np.where(energy_novelty > 0)[0][index]
    # # print(local_maximum)
    #
    # index = [[i + j for j in range(5)] for i in local_maximum]
    # index = np.array(index).reshape(-1)
    # chroma1 = librosa.feature.chroma_stft(y=wav_data, n_fft=2048, hop_length=512)[:, index]


    # current best: 12, 3, 3
    wav_data = smoothing_downsampling2(wav_data, filter_length=1000, downsampling_factor=4)
    onset_frames = librosa.onset.onset_detect(y=wav_data)[:]
    index = [[i + j + 1 for j in range(1)] for i in onset_frames]
    index = np.array(index).reshape(-1)
    index = np.clip(index, a_min=0, a_max=(wav_data.shape[0]-2048)//512)
    chroma1 = librosa.feature.chroma_stft(y=wav_data, n_fft=2048, hop_length=512)[:, :]

    chroma2 = librosa.feature.chroma_cens(y=wav_data, hop_length=512)[:, :]
    chroma3 = librosa.feature.chroma_cqt(y=wav_data, hop_length=512)[:, :]

    # chroma1 = smoothing_downsampling(chroma1, filter_length=3, downsampling_factor=3)
    # print(chroma1.shape, chroma2.shape, chroma3.shape)
    # chroma2 = smoothing_downsampling(chroma2, filter_length=3, downsampling_factor=3)
    # chroma3 = smoothing_downsampling(chroma3, filter_length=10, downsampling_factor=10)

    if vis:
        plot_chroma_vertical(chroma1, title='stft')
        plot_chroma_vertical(chroma2, title='cens')
        plot_chroma_vertical(chroma3, title='cqt')


        plt.show()
    return (chroma2 + chroma1) / 2

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