from dynamic_time_warping import *
from audio_fingerprint import *

data_root = ".\data\IOACAS_QBH_Coprus"
epislon = 1e-5

# uncomment any one of the below to select that method
# mode = 'cross'
# mode = 'sub dtw chroma'
# mode = 'sub dtw chroma norm'
mode = 'slicing dtw'
# mode = 'fingerprint spec'

data = IOACAS_dataset(data_root=data_root)
# print(len(data.chromagram_list))
data_list = data.chromagram_list

for i in range(len(data_list)):
    data_list[i] = data_list[i] / (np.max(data_list[i], axis=1).reshape(-1, 1) + epislon)

query_list = data.wav_list

max_score = -9999
target_ranking = []
for i, query in enumerate(query_list[0:]):
    score_list = []
    matching = []
    back_tracking = []
    target_score = 0
    target_ind = -1
    extracted_query_chroma = extract_feature(query)
    for j, db_chroma in enumerate(data_list):
        downsample_db_chroma = smoothing_downsampling(db_chroma, filter_length=80, downsampling_factor=80)

        if mode == 'sub dtw chroma':
            score = Dynamic_Time_Wrapping_subsequence_spec(extracted_query_chroma, downsample_db_chroma)
        elif mode == 'sub dtw chroma norm':
            score = Dynamic_Time_Wrapping_subsequence_norm(extracted_query_chroma, downsample_db_chroma)
        elif mode == 'cross':
            score = cross_correlation_chroma(extracted_query_chroma, downsample_db_chroma)
        elif mode == 'slicing dtw':
            score = slicing_dtw(query, data.db_wav_list[j])
        elif mode == 'fingerprint spec':
            dist_freq = 5  # kappa: neighborhood in frequency direction
            dist_time = 3  # tau: neighborhood in time direction
            X = compute_spectrogram(query)
            Y = compute_spectrogram(data.db_wav_list[j])

            cx = compute_constellation_map(X, dist_freq, dist_time)
            cy = compute_constellation_map(Y, dist_freq, dist_time)
            score = constellation_map_matching(cx, cy)

        matching.append(j)
        score_list.append(score)

        if str(data.singing_ground_truth_ID[i]) in data.midi_file_name[j]:
            target_score = score
            target_ind = j

    rank = (np.array(score_list) >= target_score).sum()
    target_ranking.append(rank)
    print(f'{i}: {rank}')

print('average ranking:', sum(target_ranking) / len(target_ranking))
print('hit rate:', np.sum(np.array(target_ranking) <= 10))

score_list = np.array(score_list)
ind = np.argpartition(score_list, -15)[-15:]
best_scores = score_list[ind]
best_matches = np.array(matching)[ind]
