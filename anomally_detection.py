import pandas as pd
from sklearn.cluster import dbscan
from preprocessing import preprocess
from chunk_anomally_detection import *

def dbscan_chunkus(chunks):
    noise = set()
    for chunk in chunks:
        # db = DBSCAN(eps=eps, min_samples=min_samples).fit(chunk)
        # noise_labels = chunk[db.labels_ == -1].index
        noise_labels = dbscan_chunk(chunk)
        noise = noise.union(noise_labels)
    return noise