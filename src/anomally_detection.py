from src.chunk_noise_detection import chunk_noise
import numpy as np
import math
import pandas as pd

def noise_indeces(chunks, mode):
    """returns the indeces of the noise in the original dataset

    Args:
        chunks (list): chunks to detect anomallies in

    Returns:
        set{index}: indeces from the original dataset that could be anomallies
    """

    if mode not in ['file', 'event']:
        raise ValueError("mode must be 'file' or 'event'")

    noise = set()
    for chunk in chunks:
        noise_labels = chunk_noise(chunk, mode)
        noise = noise.union(noise_labels)
    return noise


def detect_anomallies(data, mode='file', max_chunk_size=10_000, n_runs=5, seed=42):
    """Detects anomallies in the data

    Args:
        data (DataFrame): input dataset
        max_chunk_size (int, optional): Maximum size of chunks. Defaults to 10.
        n_runs (int, optional): Number of runs the algorithm will filter outliers. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        dict: dictionary of anomally indeces and number their occurences, the higher the number, the more probable the anomally
    """

    if mode not in ['file', 'event']:
        raise ValueError("mode must be 'file' or 'event'")

    anomallies = {}
    for run in range(n_runs):
        shuffled = data.sample(frac=1, random_state=seed+run)
        chunks = np.array_split(shuffled, math.ceil(shuffled.shape[0] / max_chunk_size))
        
        anomally_indeces = noise_indeces(chunks, mode)
        for index in anomally_indeces:
            anomallies[index] = anomallies.get(index, 0) + 1

    anomallies = pd.Series(anomallies)
    anomallies = anomallies[anomallies >= anomallies.max() // 2]
    return data.iloc[anomallies.index].copy()
