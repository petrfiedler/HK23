from sklearn.cluster import dbscan
from preprocessing import preprocess
import numpy as np

def chunk_noise(chunk, eps=0.5, min_samples=5):
    """return noise using DBSCAN

    Args:
        chunk (pd.DataFrame): _description_
        eps (float, optional): eps parameter of the DBSCAN algorithm. Defaults to 0.5 (DBSCAN default).
        min_samples (int, optional): min_samples parameter of the DBSCAN algorithm. Defaults to 5.

    Returns:
        np.array: array of indices corresponding to noise
    """

    chunk = preprocess(chunk)
    labels = dbscan(chunk, eps=eps, min_samples=min_samples)[1]
    noise_label_indices = np.array(chunk[labels == -1].index)
    return noise_label_indices
