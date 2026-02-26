
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import measure, morphology, filters, segmentation
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk, opening, closing
from skimage.measure import regionprops
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
from tqdm import tqdm
import subprocess

# Load the video
video_path = '/mnt/kimi/upload/Hela_CM30.avi'
cap = cv2.VideoCapture(video_path)

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video properties:")
print(f"  FPS: {fps}")
print(f"  Total frames: {total_frames}")
print(f"  Resolution: {width}x{height}")

# Read first few frames to analyze
frames = []
for i in range(min(30, total_frames)):
    ret, frame = cap.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

cap.release()

print(f"\nLoaded {len(frames)} frames for analysis")

# Visualize key frames
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
frame_indices = [0, 5, 10, 15, 20, 25]
for idx, ax in enumerate(axes.flat):
    if idx < len(frame_indices) and frame_indices[idx] < len(frames):
        ax.imshow(frames[frame_indices[idx]], cmap='gray')
        ax.set_title(f'Frame {frame_indices[idx]}')
        ax.axis('off')
plt.tight_layout()
plt.savefig('/mnt/kimi/output/sample_frames.png', dpi=150)
plt.show()
print("Sample frames saved")
