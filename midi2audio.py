from data import IOACAS_dataset
from data2 import MIR_dataset
import os
# from midi2audio import FluidSynth

'''
PIPELINE FOR CONVERTING DATABASE MIDIFILES TO WAV FILES
'''
mode = 1
if mode == 0:
    data_root = ".\data\IOACAS_QBH_Coprus"
    db = IOACAS_dataset(data_root=data_root)

    save_root = os.path.join(db.data_root, 'midi2audio')
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for midi_file_name in db.midi_file_name:
        id = midi_file_name.split('\\')[-1].split('.')[-2]
        cmd = f'fluidsynth {os.path.join(db.data_root, midi_file_name)} -F {os.path.join(save_root, id + ".wav")} -r 22050'
        print(cmd)
        os.system(cmd)

elif mode == 1:
    data_root = '.\data\MIR-QBSH-corpus'
    db = MIR_dataset(data_root=data_root)

    save_root = os.path.join(db.data_root, 'midi2audio')
    if not os.path.exists(save_root):
        os.mkdir(save_root)

    for midi_file_name in db.midi_file_list:
        id = midi_file_name.split('.')[-2]
        cmd = f'fluidsynth {os.path.join(db.data_root, "midiFile", midi_file_name)} -F {os.path.join(save_root, id + ".wav")} -r 22050'
        print(cmd)
        os.system(cmd)