import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from LoadClips import load_clips
from Autoencoder.AggregateScores import aggregate_anomaly_scores

def evaluate_model(model, input_cuboids, target_images):
    """
    Evaluate the model on input clips and calculate anomaly scores.

    Parameters:
        model (tf.keras.Model): Trained ConvLSTM model.
        input_cuboids (list of np.ndarray): Input cuboids of shape (frames_per_clip - 1, Nx, Ny, 1).
        target_images (list of np.ndarray): Target images of shape (Nx, Ny, 1).

    Returns:
        anomaly_scores (list of float): Anomaly scores for each clip.
    """
    anomaly_scores = []

    for i, (cuboid, target) in enumerate(zip(input_cuboids, target_images)):
        # Expand dimensions to match model's input shape (batch_size, frames, Nx, Ny, channels)
        cuboid = np.expand_dims(cuboid, axis=0)  # Shape: (1, frames_per_clip - 1, Nx, Ny, 1)

        # Predict the target frame
        predicted_frame = model.predict(cuboid)

        # Compute anomaly score (MSE between predicted and actual target)
        score = mean_squared_error(target.flatten(), predicted_frame.flatten())
        anomaly_scores.append(score)

    return anomaly_scores


def plot_anomaly_scores(anomaly_scores):
    """
    Plot the anomaly scores.

    Parameters:
        anomaly_scores (list of float): Anomaly scores for each clip.
    """
    #summed_scores = aggregate_anomaly_scores(anomaly_scores, num_crops_per_image=1)
    plt.figure(figsize=(10, 6))
    plt.plot(anomaly_scores, marker='o', linestyle='-', color='b', label='Anomaly Score')
    plt.title("Anomaly Scores per Clip")
    plt.xlabel("Clip Index")
    plt.ylabel("Anomaly Score (MSE)")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_reconstructed_vs_target(model, input_cuboids, target_images, clip_index=None):
    """
    Plots the target and reconstructed images side by side for a specified clip
    and the clip with the highest anomaly score.

    Parameters:
        model (keras.Model): The trained model for reconstruction.
        input_cuboids (np.ndarray): Array of input cuboids for prediction (shape: (num_clips, frames_per_clip-1, Nx, Ny, 1)).
        target_images (np.ndarray): Array of target images (shape: (num_clips, Nx, Ny, 1)).
        clip_index (int or None): Index of the specified clip to visualize. If None, skips this step.

    Returns:
        None
    """
    # Predict all reconstructed frames
    reconstructed_images = model.predict(input_cuboids, verbose=0)  # Shape: (num_clips, Nx, Ny, 1)

    # Calculate anomaly scores (Euclidean loss)
    anomaly_scores = np.linalg.norm(
        reconstructed_images.reshape(len(reconstructed_images), -1) - target_images.reshape(len(target_images), -1),
        axis=1
    )

    # Find the clip with the highest anomaly score
    highest_score_idx = np.argmax(anomaly_scores)

    def plot_images(target, reconstructed, title, score):
        """
        Helper function to plot target and reconstructed images side by side.
        """
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(target.squeeze(), cmap='gray')
        plt.title(f"Target Image\n{title}")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(reconstructed.squeeze(), cmap='gray')
        plt.title(f"Reconstructed Image\nScore: {score:.4f}")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Plot for the specified clip (if provided)
    if clip_index is not None:
        print(f"Plotting for specified clip index: {clip_index}")
        plot_images(
            target_images[clip_index],
            reconstructed_images[clip_index],
            title=f"Clip Index {clip_index}",
            score=anomaly_scores[clip_index],
        )


    # Plot for the clip with the highest anomaly score
    print(f"Plotting for clip with highest anomaly score (index {highest_score_idx})")
    plot_images(
        target_images[highest_score_idx],
        reconstructed_images[highest_score_idx],
        title="Highest Anomaly Score",
        score=anomaly_scores[highest_score_idx],
    )



input_dir = '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Cropped Clips'

input_cuboids, target_images = load_clips(input_dir, 32, 128, 8)

from tensorflow.keras.models import load_model
trained_model = load_model('Predicting.h5')

anomaly_scores = evaluate_model(trained_model, input_cuboids, target_images)
plot_anomaly_scores(anomaly_scores)
# Plot for clips with indices from 0 to 18
plot_reconstructed_vs_target(
        trained_model,
        input_cuboids,
        target_images,
        clip_index=13  # Specify the current index
    )