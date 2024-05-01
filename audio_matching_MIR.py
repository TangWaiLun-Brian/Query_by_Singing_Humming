import pandas as pd

from visualization import *
from data2 import MIR_dataset
from dynamic_time_warping import *
from audio_fingerprint import *
from chroma_util import *

data_root = ".\data\MIR-QBSH-corpus"

# below are modes for the 9 methods, uncomment any of them to execute
# mode = 'fingerprint spec'
# mode = 'sub dtw spec'
mode = 'fingerprint chroma'
# mode = 'cross'
# mode = 'sub dtw chroma dot'
# mode = 'sub dtw chroma norm'
# mode = 'sub dtw chroma shift'
# mode = 'sub dtw wav'
# mode = 'slicing dtw wav'

debug = False

data = MIR_dataset(data_root=data_root)
query_list = data.wav_data_list
data_list = data.db_wav_list

max_score = -9999
target_ranking = []
for i, query in enumerate(query_list):
    score_list = []
    matching = []
    back_tracking = []
    target_score = 0
    target_ind = -1
    extracted_query_chroma = extract_feature(query)
    for j, db_wav in enumerate(data_list):
        if mode == 'fingerprint spec':
            # spectro fingerprint

            # spectrogram fingerprint best param
            dis_freq = 7
            dis_time = 1
            tol_freq = 3
            tol_time = 1

            X = compute_spectrogram(smoothing_downsampling2(query, filter_length=40, downsampling_factor=1))
            Y = compute_spectrogram(db_wav)

            C_X = compute_constellation_maplib(X, dist_freq=dis_freq, dist_time=dis_time)
            C_Y = compute_constellation_maplib(Y, dist_freq=dis_freq, dist_time=dis_time)
            score = constellation_map_matchinglib(C_X, C_Y, tol_freq=tol_freq, tol_time=tol_time)
        elif mode == 'sub dtw spec':
            # spectro dtw
            X = compute_spectrogram(smoothing_downsampling2(query, filter_length=40, downsampling_factor=1))
            Y = compute_spectrogram(db_wav)
            score = Dynamic_Time_Wrapping_subsequence_spec(X, Y)

        elif mode == 'fingerprint chroma':
            # chroma fingerprint param
            dis_freq = 3
            dis_time = 1
            tol_freq = 1
            tol_time = 1

            downsample_db_chroma = data.chroma_list[j][:, ::4]
            # downsample_db_chroma = extract_feature(db_wav)[:, :]

            # chroma fingerprint, binary mask
            C_X = compute_constellation_map(extracted_query_chroma, dist_freq=dis_freq, dist_time=dis_time)
            C_Y = compute_constellation_map(downsample_db_chroma, dist_freq=dis_freq, dist_time=dis_time)
            score = chroma_constellation_map_matching(C_X, C_Y, tol_freq=tol_freq, tol_time=tol_time)
            # score = chroma_matching_binary(C_X, C_Y, tol_freq=tol_freq, tol_time=tol_time)

        elif mode == 'cross':
            downsample_db_chroma = data.chroma_list[j][:, ::4]
            score = cross_correlation_chroma(extracted_query_chroma, downsample_db_chroma)

        elif mode == 'sub dtw chroma dot':
            downsample_db_chroma = data.chroma_list[j][:, ::4]
            score = Dynamic_Time_Wrapping_subsequence_dot(extracted_query_chroma, downsample_db_chroma)

        elif mode == 'sub dtw chroma norm':
            downsample_db_chroma = data.chroma_list[j][:, ::4]
            score = Dynamic_Time_Wrapping_subsequence_norm(extracted_query_chroma, downsample_db_chroma)

        elif mode == 'sub dtw chroma shift':
            downsample_db_chroma = data.chroma_list[j][:, ::4]
            score = chroma_matching_dtw_subsequence_shift(extracted_query_chroma, downsample_db_chroma)

        elif mode == 'sub dtw wav':
            # wav dtw
            score = Dynamic_Time_Wrapping_subsequence_wav(query, db_wav)

        elif mode == 'slicing dtw wav':
            score = slicing_dtw(query, db_wav)

        score_list.append(score)

        if (data.song_codes[i]) == (data.df[0][j]):
            # record the gt score
            target_score = score
            target_ind = j
            if debug and 'spec' in mode:
                plot_spec_2(X, Y)

    # compute ranking for gt database file
    rank = (np.array(score_list) >= target_score).sum()
    target_ranking.append(rank)

    # compute top 10 candidate list
    ind = np.argpartition(np.array(score_list), -10)[-10:]
    df1 = data.df.iloc[ind]
    print(f'Query {i}: Rank {rank}')
    df1 = df1.set_index(pd.Index(range(1, 11)))
    df1 = df1.iloc[:, df1.columns[1]:df1.columns[3]]
    print(df1)

    if debug:
        tmp = input('Press any keys and enter to continue: ')

print('average ranking:', sum(target_ranking) / len(target_ranking))
print('hit rate:', np.sum(np.array(target_ranking) <= 10) / len(target_ranking))


