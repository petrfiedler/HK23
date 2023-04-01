from chunk_noise_detection import chunk_noise

def detect_anomallies(chunks):
    noise = set()
    for chunk in chunks:
        noise_labels = chunk_noise(chunk)
        noise = noise.union(noise_labels)
    return noise
