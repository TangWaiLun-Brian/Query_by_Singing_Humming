import os
import pretty_midi as pm
import numpy as np
import librosa

class IOACAS_dataset():
    def __init__(self, data_root=None):
        self.data_root = os.path.join(data_root, "IOACAS_pt1")
        with open(os.path.join(self.data_root, "midi.list"), 'r', encoding='gb2312') as fp:
            lines = fp.readlines()
            lines = [line.strip().split('\t') for line in lines]
            self.midi_file_name = [line[0] for line in lines]
            self.midi_song_name = [line[1] for line in lines]

        with open(os.path.join(self.data_root, "query.list"), 'r', encoding='gb2312') as fp:
            lines2 = fp.readlines()
            lines2 = [line.strip().split('\t') for line in lines2]
            self.singing_file_path = [line[0]for line in lines2]
            self.singing_ground_truth_ID = [line[1] for line in lines2]

        self.chromagram_list = []
        self.wav_list = []
        self.process_midi_files(save_file_path=os.path.join(data_root, '..', 'IOACAS_midi.npy'))
        self.process_input_wav_files(save_file_path=os.path.join(data_root, '..', 'IOACAS_wav.npy'))

    def process_input_wav_files(self, save_file_path=None):
        if save_file_path is not None and os.path.exists(save_file_path):
            print('loading preloaded wav files...')
            wav_dict = np.load(save_file_path, allow_pickle=True).item()
            self.wav_list = wav_dict.get('x')
            for i in range(len(self.wav_list)):
                self.wav_list[i] = self.wav_list[i][0]
        else:
            print('loading wav files...')
            x_list = []
            for singing_file_path in self.singing_file_path:
                x = librosa.load(os.path.join(self.data_root, singing_file_path))
                x_list.append(x)
            self.wav_list = x_list
            wav_dict = {'x': self.wav_list}
            np.save(save_file_path, wav_dict)

        print(f'total number of queries: {len(self.wav_list)}')

    def process_midi_files(self, save_file_path=None):
        if save_file_path is not None and os.path.exists(save_file_path):
            print('loading precomputed chromagram of midi file...')
            midi_dict = np.load(save_file_path, allow_pickle=True).item()
            self.chromagram_list = midi_dict.get('chromagram')
        else:
            print('computing chromagram of midi file...')
            chromagram_list = []
            for midi_file_name in self.midi_file_name:
                mid = pm.PrettyMIDI(os.path.join(self.data_root, midi_file_name))
                chromagram = mid.get_chroma(fs=22050//512)
                chromagram_list.append(chromagram)

            self.chromagram_list = chromagram_list
            midi_dict = {'chromagram': self.chromagram_list}
            np.save(save_file_path, midi_dict)

        print(f'total number of files in database: {len(self.chromagram_list)}')


# data_root = ".\data\IOACAS_QBH_Coprus"
# IOACAS_dataset(data_root=data_root)