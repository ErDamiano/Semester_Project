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
    regularity_scores = [
        np.mean((original - reconstructed) ** 2)
        for original, reconstructed in zip(target_images, reconstructed_images)
    ]

    # Plot regularity scores
    plt.figure(figsize=(10, 6))
    plt.plot(regularity_scores, marker='o', linestyle='-', color='b')
    plt.title('Anomaly Scores for Reconstructed Images')
    plt.xlabel('Image Index')
    plt.ylabel('Anomaly Score (MSE)')
    plt.grid()
    plt.show()


    # Specify the index of the image you want to display
    specified_index = 10  # Change this to the index of your desired image

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


    return regularity_scores

# Load the trained model
from tensorflow.keras.models import load_model
autoencoder = load_model('denoising_autoencoder.h5')

# Evaluate and plot regularity scores
evaluate_autoencoder(autoencoder,
                     '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/All processed images',
                     '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/AugmentedFrames')






