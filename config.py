# Configuration for Cell Segmentation Task
import os

# Paths
VIDEO_INPUT_PATH = r"C:\Users\Admin\cellpose-sample\sample.avi.avi"
OUTPUT_DIR = r"C:\Users\Admin\cellpose-sample\results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Cellpose Model Settings
MODEL_TYPE = "nuclei"  # Options: 'nuclei', 'cyto', 'cyto2'
GPU_AVAILABLE = False  # Set to True if you have CUDA
DIAMETER = 30  # Average cell diameter in pixels (adjust based on your data)

# Cell Classification Thresholds
# Frame ranges for classifying cells
EARLY_FRAMES = (0, 50)  # Circular cells before fixation
MID_FRAMES = (51, 150)  # Fixed cells
LATE_FRAMES = (151, float('inf'))  # Dead circular cells after fixation

# Fragment detection (based on area)
MIN_CELL_AREA = 50  # pixels²
MAX_CELL_AREA = 10000  # pixels²
FRAGMENT_MAX_AREA = 50  # pixels² - Objects smaller than this are fragments

# Output settings
VIDEO_FPS = 30  # Frames per second for output video
CODEC = 'MJPG'  # Video codec