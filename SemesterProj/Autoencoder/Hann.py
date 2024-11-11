import numpy as np


def smooth_anomaly_scores(scores, window_size=5):
    """
    Smooth the anomaly scores using a Hann window.

    Parameters:
    scores (list or numpy array): List of anomaly scores to be smoothed.
    window_size (int): Size of the Hann window (must be an odd number).

    Returns:
    numpy array: Smoothed anomaly scores.
    """
    # Generate a Hann window of the specified size
    hann_window = np.hanning(window_size)

    # Normalize the Hann window so the sum is 1
    hann_window /= hann_window.sum()

    # Apply convolution between scores and the Hann window
    smoothed_scores = np.convolve(scores, hann_window, mode='same')

    return smoothed_scores