from ChoseWindow import choose_window

# Set initial parameters
Nx = 512  # Default width window
Ny = 128  # Default height window
x, y = 1100, 1000  # Default center coordinates for the crop window

# Input and output directories for MAC
input_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_903"
output_dir = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/croppedFrames/MN_903"

# Call the function with parameters
choose_window(input_dir, output_dir, x, y, Nx, Ny)