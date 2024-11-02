import numpy as np
import cv2
import os
import glob
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


# Load images
def load_images(inputDir):
    images = []
    for file in sorted(glob.glob(f"{inputDir}/*.jpg")):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 128))  # Ensure all images are the same size
        images.append(img)
    return np.array(images)


# Add Gaussian noise
def add_gaussian_noise(images, mean=0, sigma=16, patch_size=16, num_patches=10):
    noisy_images = []
    for img in images:
        img_height, img_width = img.shape
        noisy_img = img.copy()

        for _ in range(num_patches):
            # Randomly select the top-left corner for the patch
            x = np.random.randint(0, img_width - patch_size)
            y = np.random.randint(0, img_height - patch_size)

            # Generate 16x16 Gaussian noise patch
            noise_patch = np.random.normal(mean, sigma, (patch_size, patch_size))

            # Add the noise patch to the selected region
            noisy_img[y:y + patch_size, x:x + patch_size] += noise_patch

        # Clip to ensure pixel values remain valid
        noisy_img = np.clip(noisy_img, 0, 255)
        noisy_images.append(noisy_img)

    return np.array(noisy_images).astype(np.uint8)


# Build the autoencoder model
def build_autoencoder():
    model = models.Sequential()
    model.add(layers.Input(shape=(128, 256, 1)))  # Adjust according to your input shape
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same'))  # Output layer
    return model


# Load and prepare data
input_dir = '/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/All_Augmented'  # Update with your path
images = load_images(input_dir)
images = images.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Add noise
noisy_images = add_gaussian_noise(images)

# Split data into training and validation sets before reshaping
X_train, X_val, y_train, y_val = train_test_split(noisy_images, images, test_size=0.2, random_state=42)

# Reshape for the model (if needed)
X_train = X_train.reshape(-1, 128, 256, 1)
X_val = X_val.reshape(-1, 128, 256, 1)
y_train = y_train.reshape(-1, 128, 256, 1)
y_val = y_val.reshape(-1, 128, 256, 1)

# Build and compile the autoencoder
autoencoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model and calculate anomaly scores
reconstructed_images = autoencoder.predict(X_val)
mse = np.mean(np.square(reconstructed_images - images.reshape(-1, 128, 256, 1)), axis=(1, 2, 3))

# Anomaly scores
anomaly_scores = mse

# Save or display your results as needed

