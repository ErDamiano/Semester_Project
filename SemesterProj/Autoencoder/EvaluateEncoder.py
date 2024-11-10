import numpy as np
import matplotlib.pyplot as plt
import os
from LoadInputAndTarget import load_images

def evaluate_autoencoder(autoencoder, input_dir, target_dir, num_samples=5):
    # Load images
    input_images, target_images = load_images(input_dir, target_dir)

    # List all files in the target directory
    files = os.listdir(target_dir)
    files = sorted(files, key=lambda x: int(os.path.splitext(x)[0]))  # Numeric sorting

    # Ensure the input and target images are in the same order
    assert len(input_images) == len(target_images), "Number of input and target images must match"

    # Get reconstructed images from the autoencoder
    reconstructed_images = autoencoder.predict(input_images)

    # Calculate regularity scores (Mean Squared Error)
    anomaly_scores = [
        np.mean((original - reconstructed) ** 2)
        for original, reconstructed in zip(target_images, reconstructed_images)
    ]

    top_n = 20  # Number of top scores to retrieve
    top_indices = np.argsort(anomaly_scores)[-top_n:][::-1]  # Sort and reverse for descending order

    # Print top indices and their corresponding scores
    for idx in top_indices:
        print(f"Index: {idx}, Anomaly Score: {anomaly_scores[idx]}")

    x_values = np.linspace(0, len(anomaly_scores), len(anomaly_scores))

    # Set the backend to TkAgg to ensure an interactive window
    plt.switch_backend('TkAgg')

    # Plot regularity scores
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.plot(anomaly_scores, marker='o', linestyle='-', color='b')
    plt.title('Anomaly Scores for Reconstructed Images')
    plt.xlabel('Image Index')
    plt.ylabel('Anomaly Score (MSE)')
    plt.grid()

    def on_click(event):
        if event.inaxes == ax:
            x, y = event.xdata, event.ydata
            print(f"Clicked at x={x:.0f}, y={y:.4f}")

    # Connect the event to the figure
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Specify the index of the image you want to display
    specified_index = 176  # Change this to the index of your desired image

    # Create a single figure for both sets of images
    plt.figure(figsize=(12, 8))

    plt.subplot(1, 3, 1)
    plt.imshow(target_images[specified_index].squeeze(), cmap='gray')
    plt.title(f"Original Image {specified_index}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(input_images[specified_index].squeeze(), cmap='gray')
    plt.title(f"Noisy Input Image {specified_index}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(reconstructed_images[specified_index].squeeze(), cmap='gray')
    plt.title(f"Reconstructed Image {specified_index}")
    plt.axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

    return anomaly_scores

# Load the trained model
from tensorflow.keras.models import load_model
autoencoder = load_model('denoising_autoencoder.h5')

# Evaluate and plot regularity scores
evaluate_autoencoder(autoencoder,
                     '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/All processed images',
                     '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/AugmentedFrames')






