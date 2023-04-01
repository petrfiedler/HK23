from sklearn.cluster import dbscan
from preprocessing import preprocess

def chunk_noise(chunk, eps, min_samples):
    chunk = preprocess(chunk)
    labels = dbscan(chunk, eps=eps, min_samples=min_samples)[1]
    noise_label_indices = chunk[labels == -1].index
    return noise_label_indices
