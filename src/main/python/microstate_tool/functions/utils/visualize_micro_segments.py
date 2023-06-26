
from dyconnmap.fc.plv import plv
from dyconnmap.fc.coherence import coherence
from dyconnmap.fc.corr import corr
from dyconnmap.fc.cos import cos
from dyconnmap.fc.icoherence import icoherence
from dyconnmap.fc.iplv import iplv
from dyconnmap.fc.pli import pli

def estimate_fc_mat(eeg_data, method, channel_names):

    fs = 250
    fb = [2, 20]

    if method == 'COH':
        ### Coherence
        CONN_MAT = coherence(eeg_data, fb, fs)
    elif method == 'CORR':
        ### Correlation
        CONN_MAT = corr(eeg_data, fb, fs)
    elif method == 'COS':
        ### Cosine
        CONN_MAT = cos(eeg_data, fb, fs)
    elif method == 'ICOH':
        ### Imaginary Coherence
        CONN_MAT = icoherence(eeg_data, fb, fs)
    elif method == 'PLV':
        ### Phase Locking Value
        CONN_MAT = plv(eeg_data, fb, fs)[1]
    elif method == 'IPLV':
        ### Imaginary part of Phase Locking Value
        CONN_MAT = iplv(eeg_data, fb, fs)[1]
    elif method == 'PLI':
        ### Phase Lag Index
        CONN_MAT = pli(eeg_data, fb, fs)[1]
    else:
        raise ValueError("Failed to match method")

    CONN_MAT = CONN_MAT + CONN_MAT.T
    return CONN_MAT