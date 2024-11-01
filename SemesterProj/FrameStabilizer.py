import cv2
import numpy as np
import glob

# Load frames Mac
frames = [cv2.imread(file) for file in sorted(glob.glob("/Volumes/MEMORIAEST/SemesterProjectData/2024_01_08E/MN_00/2024_01_08E_MN_00_001/pics/2D/*.tiff"))]
# Load frames Windows
# frames = [cv2.imread(file) for file in sorted(glob.glob("G:\2D\2024_01_08E_PU_00_001-0001_0.tiff"))]

# Convert frames to grayscale for better feature detection
gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]

# Define the first frame as the reference
reference_frame = gray_frames[0]
height, width = reference_frame.shape

# Create an empty list to hold stabilized frames
stabilized_frames = []
k = 0

# Loop through each frame and align to the reference
for frame in gray_frames:
    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(reference_frame, None)
    kp2, des2 = orb.detectAndCompute(frame, None)

    # Match features between the reference frame and current frame
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched keypoints
    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Estimate transformation matrix
    matrix, mask = cv2.estimateAffinePartial2D(pts2, pts1)

    # Apply affine transformation to stabilize the frame
    stabilized_frame = cv2.warpAffine(frame, matrix, (width, height))

    # Append stabilized frame to the list
    stabilized_frames.append(stabilized_frame)

# Save or display stabilized frames Mac
output_folder = "/Users/DamianFrei/Desktop/ETH/Master/SemesterProject/stabilizedFrames/MN_001/"

# Save or display stabilized frames Windows
# output_folder = "C:\Users\damfrei\Desktop\SemesterProject\SemesterProj\StabilizedFrames"

for i, frame in enumerate(stabilized_frames):
    cv2.imwrite(f"{output_folder}stabilized_frame_{i}.jpg", frame)
    k=k+1
    print(k)