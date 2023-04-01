from chunk_noise_detection import chunk_noise
import numpy as np
import math

def noise_indeces(chunks):
    """returns the indeces of the noise in the original dataset

    Args:
        chunks (list): chunks to detect anomallies in

    Returns:
        set{index}: indeces from the original dataset that could be anomallies
    """

    noise = set()
    for chunk in chunks:
        noise_labels = chunk_noise(chunk)
        noise = noise.union(noise_labels)
    return noise


def detect_anomallies(data, max_chunk_size=10_000, n_runs=5, seed=42):
    """Detects anomallies in the data

    Args:
        data (DataFrame): input dataset
        max_chunk_size (int, optional): Maximum size of chunks. Defaults to 10.
        n_runs (int, optional): Number of runs the algorithm will filter outliers. Defaults to 5.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        dict: dictionary of anomally indeces and number their occurences, the higher the number, the more probable the anomally
    """

    anomallies = {}
    for run in range(n_runs):
        shuffled = data.sample(frac=1, random_state=seed+run)
        chunks = np.array_split(shuffled, math.ceil(shuffled.shape[0] / max_chunk_size))
        
        anomally_indeces = noise_indeces(chunks)
        for index in anomally_indeces:
            anomallies[index] = anomallies.get(index, 0) + 1
    return anomallies
