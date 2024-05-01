import libfmp.b
import libfmp.c4
import libfmp.c7
import scipy

from visualization import plot_chroma_vertical
from data import IOACAS_dataset
from preprocessing import *

def compute_cost_matrix(query, db):
    '''
    utility function for cost matrix computation
    :param query: query wave data
    :param db: database wave data
    :return: cost matrix
    '''
    return libfmp.c3.compute_cost_matrix(query, db, metric='seuclidean')


def compute_accumulated_cost_matrix(C):
    '''
    utility function for accumulated cost matrix computation
    Reference: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C7/C7S2_SubsequenceDTW.html
    :param C: cost matrix
    :return: accumulated cost matrix
    '''
    n = C.shape[0]
    m = C.shape[1]

    D = np.zeros((n + 2, m + 2))
    D[:2, :] = np.inf
    D[:, :2] = np.inf
    D[2, 2] = C[0, 0]

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            D[i + 2, j + 2] = min(D[i + 1, j + 0], D[i + 0, j + 1], D[i + 1, j + 1]) + C[i, j]

    return D[2:, 2:]


def compute_optimal_warping_path(D):
    i = D.shape[0] - 1
    j = D.shape[1] - 1
    P = []
    P.append((i, j))

    while i > 0 or j > 0:
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            min_val = min(D[i - 2, j - 1], D[i - 1, j - 2], D[i - 1, j - 1])
            if min_val == D[i - 1, j - 1]:
                i -= 1
                j -= 1
            elif min_val == D[i - 1, j - 2]:
                i -= 1
                j -= 2
            else:
                i -= 2
                j -= 1

        P.append((i, j))

    P.reverse()
    P = np.array(P)
    return P

def slicing_dtw(q, y, downsample_rate=30000, sr=22050):
    '''
    perform slicing dtw
    :param q: query wave data
    :param y: database wave data
    :param downsample_rate: downsampling rate on both audio
    :param sr: sampling rate
    :return: matching score
    '''
    # downsample_rate = 5000
    # q, sr = librosa.load('001_002.wav')
    X = smoothing_downsampling2(q, filter_length=250, downsampling_factor=downsample_rate, kernel_type='boxcar')

    # y, sr = librosa.load('10027.wav')
    db = y / np.max(np.abs(y))

    # db_filter = db[np.abs(db) >= 1e-2]
    db_filter = db
    onsets_times = librosa.onset.onset_detect(y=db_filter, sr=sr, units='time')
    onsets_sindex = (onsets_times * sr // downsample_rate).astype(int)
    L = onsets_sindex.shape[0]
    N = X.shape[0]
    i = 0
    score = 9999999
    P_opt = []
    sindex_opt = -1
    # db_downsample = db_filter[::downsample_rate]
    db_downsample = smoothing_downsampling2(db_filter, filter_length=250, downsampling_factor=downsample_rate, kernel_type='boxcar')

    M = db_downsample.shape[0]

    while i < L and onsets_sindex[i] + N < M:
        A = X
        B = db_downsample[onsets_sindex[i]: onsets_sindex[i] + N]
        C = libfmp.c3.compute_cost_matrix(A, B, metric='seuclidean')
        D = compute_accumulated_cost_matrix(C)
        P = compute_optimal_warping_path(D)
        # print(D[-1, -1])
        if D[-1, -1] < score:
            score = D[-1, -1]
            P_opt = P
            sindex_opt = onsets_sindex[i]
            C_opt = C
            D_opt = D
            B_opt = B
        i += 1
    return -score

def Dynamic_Time_Wrapping_subsequence_spec(query_spec, db_spec):
    '''
    utility function for subsequence dtw on spectrogram
    :param query_spec: query spectrogram
    :param db_spec: database spectrogram
    :return: mathcing score
    '''
    C = libfmp.c7.cost_matrix_dot(query_spec, db_spec)
    D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw(C)
    length = db_spec.shape[1]

    return -D[-1, -1] / length

def Dynamic_Time_Wrapping_subsequence_wav(query, db):
    '''
    utility function for subsequence dtw on wave data
    :param query: query wave data
    :param db: database wave data
    :return: matching score
    '''
    # C = libfmp.c7.cost_matrix_dot(query, db)
    query = smoothing_downsampling2(query, filter_length=300, downsampling_factor=15000)
    db = smoothing_downsampling2(db, filter_length=300, downsampling_factor=15000)

    C = scipy.spatial.distance_matrix(query.reshape(-1, 1), db.reshape(-1, 1))
    D = libfmp.c7.compute_accumulated_cost_matrix_subsequence_dtw_21(C)
    length = db.shape[0]

    return -D[-1, -1] / length

def cross_correlation_chroma(query_chroma, db_chroma):
    '''
    perform cross correlation on chroma features
    :param query_chroma: query chroma features
    :param db_chroma: database chroma features
    :return: matching score
    '''
    n = db_chroma.shape[1] - query_chroma.shape[1] + 1
    length = query_chroma.shape[1]
    max_score = -9999
    for i in range(n):
        score = db_chroma[:, i:i+length].reshape(-1) @ query_chroma[:, :].reshape(-1)
        max_score = max(max_score, score)

    return max_score

def Dynamic_Time_Wrapping_subsequence_dot(query_chroma, db_chroma):
    '''
    perform subsequence dtw on chroma with dot product
    :param query_chroma: query chroma
    :param db_chroma: database chroma
    :return: mathcing score
    '''
    n = db_chroma.shape[1]
    m = query_chroma.shape[1]
    Match_Matrix = np.zeros((n+1, m+1))
    Back_Tracking_Matrix = np.zeros((n+1, m+1))
    Track_dir = [-1, -m-2, -m-1]
    for i in range(1, n+1):
        for j in range(1, m+1):
            # cost = np.linalg.norm(db_chroma[:, i - 1] - query_chroma[:, j - 1]) - np.sqrt(12)
            cost = -(db_chroma[:, i - 1] @ query_chroma[:, j - 1])
            dependency = np.array([Match_Matrix[i, j-1], Match_Matrix[i-1, j-1] + cost, Match_Matrix[i-1, j]])
            index = np.argmin(dependency)
            Match_Matrix[i, j] = dependency[index]
            Back_Tracking_Matrix[i, j] = Track_dir[index]


    index = (n+1) * (m+1) - 1
    i = int(index) // (m+1)
    j = int(index) % (m+1)
    subsequence = []
    while Back_Tracking_Matrix[i, j] != 0:
        # print(i, j, Back_Tracking_Matrix[i, j])
        subsequence.append(index)
        index += Back_Tracking_Matrix[i, j]
        i = int(index) // (m + 1)
        j = int(index) % (m + 1)

    subsequence = np.array(subsequence)
    horizontal = subsequence % (m+1)
    vertical = subsequence // (m+1)
    vertical_spacing = vertical[1:] - vertical[:-1]
    penalty = np.abs(vertical_spacing[1:] - vertical_spacing[:-1]) ** 2 * 2
    return -(Match_Matrix[n, m] + np.sum(penalty))

def Dynamic_Time_Wrapping_subsequence_norm(query_chroma, db_chroma):
    '''
    perform subsequence dtw on chroma with euclidean distance
    :param query_chroma: query chroma
    :param db_chroma: database chroma
    :return: matching score
    '''
    n = db_chroma.shape[1]
    m = query_chroma.shape[1]
    Match_Matrix = np.zeros((n+1, m+1))
    Back_Tracking_Matrix = np.zeros((n+1, m+1))
    Track_dir = [-1, -m-2, -m-1]
    for i in range(1, m+1):
        Match_Matrix[1, i] = Match_Matrix[1, i-1] + np.linalg.norm(query_chroma[:, i-1] - db_chroma[:, 0])
    for j in range(1, n+1):
        Match_Matrix[j, 1] = np.linalg.norm(query_chroma[:, 0] - db_chroma[:, j-1])

    for i in range(2, n+1):
        for j in range(2, m+0):
            dependency = np.array([Match_Matrix[i, j-1], Match_Matrix[i-1, j-1], Match_Matrix[i-1, j]])
            index = np.argmin(dependency)
            Match_Matrix[i, j] = dependency[index] + np.linalg.norm(db_chroma[:, i - 1] - query_chroma[:, j - 1])
            Back_Tracking_Matrix[i, j] = Track_dir[index]

        dependency = np.array([Match_Matrix[i, m - 1], Match_Matrix[i - 1, m - 1], Match_Matrix[i - 1, m]])
        index = np.argmin(dependency)
        Match_Matrix[i, m] = dependency[index]
        if index != 2:
            Match_Matrix[i, m] += np.linalg.norm(db_chroma[:, i - 1] - query_chroma[:, m - 1])
        Back_Tracking_Matrix[i, m] = Track_dir[index]

    index = (n+1) * (m+1) - 1
    i = int(index) // (m+1)
    j = int(index) % (m+1)
    subsequence = []
    while Back_Tracking_Matrix[i, j] != 0:
        # print(i, j, Back_Tracking_Matrix[i, j])
        subsequence.append(index)
        index += Back_Tracking_Matrix[i, j]
        i = int(index) // (m + 1)
        j = int(index) % (m + 1)

    return -Match_Matrix[n, m]

