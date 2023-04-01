from chunk_noise_detection import chunk_noise

def detect_anomallies(chunks):
    """detects anomallies in a list of chunks

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
