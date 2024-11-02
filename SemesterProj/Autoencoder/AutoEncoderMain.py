from ChooseWindow import choose_window
from DataAugmentation import data_augmentation
from main import output_dir

# Set initial parameters to crop the images around x y with a window size of Nx and Ny
Nx = 512  # Default width window
Ny = 128  # Default height window
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