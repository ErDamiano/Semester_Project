import matplotlib.pyplot as plt
import numpy as np

# Sample anomaly scores for demonstration
anomaly_scores = np.random.rand(20)  # Replace with your actual anomaly score list

# Set the backend to TkAgg to ensure an interactive window
plt.switch_backend('TkAgg')

# Plot regularity scores
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.plot(anomaly_scores, marker='o', linestyle='-', color='b')
plt.title('Anomaly Scores for Reconstructed Images')
plt.xlabel('Image Index')
plt.ylabel('Anomaly Score (MSE)')
plt.grid()

# Click event function
def on_click(event):
    if event.inaxes == ax:
        x, y = event.xdata, event.ydata
        print(f"Clicked at x={x:.0f}, y={y:.4f}")

# Connect the event to the figure
fig.canvas.mpl_connect('button_press_event', on_click)

# Display the plot
plt.show()