import numpy as np

def aggregate_anomaly_scores(anomaly_scores, num_crops_per_image=5):

    # Reshape the array so each row represents scores for one image's cropped frames
    scores_per_image = np.array(anomaly_scores).reshape(-1, num_crops_per_image)

    # Sum the scores for each set of crops to get a single score per image
    aggregated_scores = scores_per_image.sum(axis=1)

    return aggregated_scores