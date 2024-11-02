from ChooseWindow import choose_window
from DataAugmentation import data_augmentation
from GaussianNoise import apply_noise_to_directory

# Set initial parameters to crop the images around x y with a window size of Nx and Ny
Nx = 512  # Default width window
Ny = 512  # Default height window
x, y = 1100, 1000  # Default center coordinates for the crop window

# Input and output directories for MAC
input_dir_window_size = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Test"
output_dir_window_size = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Test/Cropped"

# Choose the window size to be cropped out of the stabilized images
choose_window(input_dir_window_size, output_dir_window_size, x, y, Nx, Ny)

input_dir_augment = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Test/Cropped"
output_dir_augment = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Test/Augmented"
# Call the Data Augmentation function to make the frames more different from each other
# and avoid overfitting
data_augmentation(input_dir_augment, output_dir_augment)

input_dir_noisy = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Test/Augmented"
output_dir_noisy = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/Test/Noisy"
apply_noise_to_directory(input_dir_noisy, output_dir_noisy, 8, 0.4)
