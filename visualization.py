import numpy as np
import librosa
import matplotlib.pyplot as plt

def plot_chroma_vertical(chromagram, title=''):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(chromagram, y_axis='chroma', x_axis='time', ax=ax)
    ax.set(title=title)
    fig.colorbar(img, ax=ax)