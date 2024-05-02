Project Title:
Query by singing/humming in MIREX2021

Description:
A Python program created for the AIST3110 course project, by Stephen and Brian. It is a model that allows users to
test on two datasets, MIR-QBSH and IOACAS. By inputting the query audios, the program will try to match it with one
of the ground-truth audios in the database, and return a top-10 candidate list according to how likely the songs
are to be the desired one.

Dataset:
We mainly use MIR-QBSH-corpus as our dataset.
It can be downloaded via https://music-ir.org/evaluation/MIREX/data/qbsh/MIR-QBSH-corpus.tar.gz
Please make sure it is located at '.\data\MIR-QBSH-corpus\'.

If you wish to test on IOACAS-corpus, it can be downloaded via https://music-ir.org/evaluation/MIREX/data/qbsh/IOACAS_QBH_Corpus.tar.gz
Please make sure it is located at '.\data\'IOACAS_QBH_Coprus'.

Installation:
This project requires the use of Python, please set up a virtual environment with Anaconda/Miniconda or python with
virtualenv.
After you have the virtual environment, install the packages using pip and run the following command:
pip install -r requirements.txt


Preprocessing the ground-truth MIDI files:
If you don't have the ground-truth audios in audio format, run the 'midi2audio.py' file, this will convert the
MIDI files into .wav format. Make sure the audios are placed under '.\data\MIR-QBSH-corpus\midi2audio'.
Adjust mode to perform the same task for IOACAS-corpus.


Usage:
To test the first 50 query audios, simply run 'audio_matching_MIR.py'.
There are more settings to test with, for instance, you can uncomment any of the 9 methods under the same file.
Available methods:
- Audio fingerprinting with spectrogram
- Subsequence DTW with spectrogram
- Audio fingerprinting with chroma
- Cross correlation
- Subsequence DTW with chroma dot product
- Subsequence DTW with chroma Euclidean norm
- Subsequence DTW with chroma shift
- Subsequence DTW with waveform
- DTW with slicing technique
Moreover, if you wish to pause the process after each query, you can set the debug variable to True. This will prompt
a user input and will only continue when you press any keys and enter.

Similarly, test can be performed on IOACAS-corpus by running 'audio_matching_IOACAS.py'.