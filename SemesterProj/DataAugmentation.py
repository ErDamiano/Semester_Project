import cv2
import glob
import imgaug.augmenters as iaa
import numpy as np

# Load images
input_dir = [
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_001"  # Adjust the path as necessary
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_002"
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_003"
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_451"
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_452"
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_453"
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_901"
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_902"
    "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_903"
    ]
images = [cv2.imread(file, cv2.IMREAD_GRAYSCALE) for file in glob.glob(input_dir)]

# Convert images to a format suitable for imgaug
images = np.array(images)

# Define an augmentation sequence
seq = iaa.Sequential([
    iaa.Fliplr(0.5),  # Flip images horizontally with 50% probability
    iaa.Affine(
        rotate=(-10, 10),  # Rotate images by -10 to +10 degrees
        scale=(0.9, 1.1)  # Scale images to 90% to 110%
    ),
    iaa.AdditiveGaussianNoise(scale=(0, 0.1 * 255)),  # Add Gaussian noise
    iaa.ContrastNormalization((0.8, 1.2)),  # Adjust contrast
    iaa.Crop(percent=(0, 0.1)),  # Randomly crop images
])

# Apply augmentations to each image
augmented_images = seq(images=images)

# Save or use augmented images
output_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/All_Augmented/"
for i, img in enumerate(augmented_images):
    cv2.imwrite(f"{output_dir}/augmented_image_{i}.jpg", img)

print("Augmentation completed and saved to:", output_dir)