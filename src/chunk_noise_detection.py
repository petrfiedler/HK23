from sklearn.cluster import dbscan
from src.preprocessing import preprocess_event, preprocess_file
import numpy as np

OPTIMAL_HYPERPARAMETERS = {
    'file': {
        'eps': 0.7,
        'min_samples': 3
    },
    'event': {
        'eps': 0.5,
        'min_samples': 3
    }
}

def chunk_noise(chunk, mode):
    """return noise using DBSCAN

    Args:
        chunk (pd.DataFrame): _description_

    Returns:
        np.array: array of indices corresponding to noise
    """

    if mode == 'file':
        chunk = preprocess_file(chunk)
    elif mode == 'event':
        chunk = preprocess_event(chunk)
    else:
        raise ValueError("mode must be 'file' or 'event'")
    
    labels = dbscan(chunk, eps=OPTIMAL_HYPERPARAMETERS[mode]['eps'], min_samples=OPTIMAL_HYPERPARAMETERS[mode]['min_samples'])[1]
    noise_label_indices = np.array(chunk[labels == -1].index)
    return noise_label_indices
