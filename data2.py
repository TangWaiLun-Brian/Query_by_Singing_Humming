import os
import librosa
import pandas as pd

class MIR_dataset():
    def __init__(self, data_root=None):
        self.data_root = data_root

        # create a songs dictionary so the song number correspond to the names of the song and the number of samples
        file_path = os.path.join(self.data_root, 'midiFile/songList.txt')
        self.df = pd.read_csv(file_path, sep='\t', header=None, encoding='Big5')
        self.df[0] = self.df[0].apply(lambda x: str(x).zfill(5))
        self.songs_dict = self.df.set_index(0).apply(lambda x: [x[1], x[2], x[3]], axis=1).to_dict()
        print(self.df)
        print(self.songs_dict)

        # create a midi file path list
        self.midi_file_list = [code + '.mid' for code in self.df[0]]
        print(self.midi_file_list)

        # load all the .wav files into a list with librosa
        wave_folder = os.path.join(self.data_root, 'waveFile')

        self.wav_files = []
        self.pv_files = []
        self.song_codes = []
        # go through all the year folders
        for year_folder in os.listdir(wave_folder):
            year_folder_path = os.path.join(wave_folder, year_folder)

            # skip if the current file is not a directory
            if not os.path.isdir(year_folder_path):
                continue

            # go through all the person folders
            for person_folder in os.listdir(year_folder_path):
                person_folder_path = os.path.join(year_folder_path, person_folder)

                # skip if the current file is not a directory
                if not os.path.isdir(person_folder_path):
                    continue

                # go through all the files in the person folder
                for file in os.listdir(person_folder_path):
                    file_path = os.path.join(person_folder_path, file)

                    # check if it's a .wav file
                    if file.lower().endswith('.wav'):
                        # store the .wav file paths
                        self.wav_files.append(file_path)
                        # extract the song code from the file name and store it
                        song_code = file.split('.')[0]
                        self.song_codes.append(song_code)

                    # check if it's a .pv file
                    if file.lower().endswith('.pv'):
                        # load the .pv file into a list
                        with open(file_path, 'r') as pv_file:
                            pv_data = [float(line.strip()) for line in pv_file]
                            self.pv_files.append(pv_data)

        # load the .wav files using librosa
        print('loading files to librosa...')
        self.wav_data_list = [librosa.load(file)[0] for file in self.wav_files]
        self.db_wav_list = [librosa.load(os.path.join(self.data_root, 'midi2audio', code + '.wav'))[0] for code in self.song_codes]
        print(f"Number of WAV Files: {len(self.wav_data_list)}")
        print(f"List of song codes: {self.song_codes}")

        # check the range of the numbers in pv_data
        min_value = min(min(file_data) for file_data in self.pv_files)
        max_value = max(max(file_data) for file_data in self.pv_files)
        print(f'Range of the values in pv_data: [{min_value}, {max_value}]')

if __name__ == '__main__':
    data_root = ".\data\MIR-QBSH-corpus"
    MIR_dataset(data_root=data_root)