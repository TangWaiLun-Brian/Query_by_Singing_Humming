import numpy as np
import librosa
import matplotlib.pyplot as plt


def plot_chroma_vertical(chromagram, title=''):
    '''
    plotting chromagrams
    :param chromagram: chroma
    :param title: title of subplot
    :return: None
    '''
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax)

def plot_spec_2(X, Y):
    '''
    plotting a pair of spectrograms
    :param X: top spectrogram
    :param Y: bottom spectrogram
    :return: None
    '''
    fig, ax = plt.subplots(2, 1)
    img = librosa.display.specshow(X, ax=ax[0])
    fig.colorbar(img, ax=ax[0])

    img = librosa.display.specshow(Y, ax=ax[1])
    fig.colorbar(img, ax=ax[1])
    plt.show()
