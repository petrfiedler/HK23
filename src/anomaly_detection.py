from src.chunk_noise_detection import chunk_noise
import numpy as np
import math
import pandas as pd

def noise_indeces(chunks, mode):
    """returns the indeces of the noise in the original dataset

    Args:
        chunks (list): chunks to detect anomalies in

    Returns:
        set{index}: indeces from the original dataset that could be anomalies
    """

    if mode not in ['file', 'event']:
        raise ValueError("mode must be 'file' or 'event'")

    noise = set()
    for chunk in chunks:
        noise_labels = chunk_noise(chunk, mode)
        noise = noise.union(noise_labels)
    return noise


def detect_anomalies(data, mode='file', max_chunk_size=10_000, n_runs=5, seed=42):
    """Detects anomalies in the data

    Args:
        data (DataFrame): input dataset
        max_chunk_size (int, optional): Maximum size of chunks. Defaults to 10.
        n_runs (int, optional): Number of runs the algorithm will filter outliers. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        dict: dictionary of anomaly indeces and number their occurences, the higher the number, the more probable the anomaly
    """

    if mode not in ['file', 'event']:
        raise ValueError("mode must be 'file' or 'event'")

    anomalies = {}
    for run in range(n_runs):
        shuffled = data.sample(frac=1, random_state=seed+run)
        chunks = np.array_split(shuffled, math.ceil(shuffled.shape[0] / max_chunk_size))
        
        anomaly_indeces = noise_indeces(chunks, mode)
        for index in anomaly_indeces:
            anomalies[index] = anomalies.get(index, 0) + 1

    anomalies = pd.Series(anomalies)
    anomalies = anomalies[anomalies >= anomalies.max() // 2]
    return data.iloc[anomalies.index].copy()
