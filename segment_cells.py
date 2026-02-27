# -*- coding: utf-8 -*-
"""
Cell Segmentation using Cellpose
Segments cells into 4 categories: Circular (early), Fixed, Dead, Fragments
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from cellpose import models
from skimage.measure import regionprops
import warnings
warnings.filterwarnings('ignore')

from config import *

def load_video(video_path):
    """Load video and return cap object with metadata"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    return cap, fps, width, height, frame_count

def initialize_model():
    """Initialize Cellpose model"""
    print("Loading Cellpose model...")
    model = models.Cellpose(
        gpu=GPU_AVAILABLE,
        model_type=MODEL_TYPE
    )
    return model

def classify_cell(frame_num, area, brightness):
    """Classify cell into 4 categories"""
    # Check if it's a fragment
    if area < FRAGMENT_MAX_AREA:
        return 'fragment', 3
    
    # Classify by frame position
    if EARLY_FRAMES[0] <= frame_num <= EARLY_FRAMES[1]:
        return 'circular', 0
    elif MID_FRAMES[0] <= frame_num <= MID_FRAMES[1]:
        return 'fixed', 1
    elif frame_num >= LATE_FRAMES[0]:
        return 'dead', 2
    else:
        return 'unknown', 3

def create_mask_visualization(mask, class_id):
    """Create colored mask visualization"""
    colors = [
        (0, 255, 0),    # Green - Circular
        (255, 255, 0),  # Cyan - Fixed
        (0, 0, 255),    # Red - Dead
        (128, 128, 128) # Gray - Fragments
    ]
    
    mask_viz = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_viz[mask > 0] = colors[class_id]
    return mask_viz

def process_video(model, cap, fps, width, height, frame_count):
    """Process entire video and segment cells"""
    
    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*CODEC)
    out_masks = cv2.VideoWriter(
        str(Path(OUTPUT_DIR) / "segmentation_masks.avi"),
        fourcc, fps, (width, height)
    )
    out_overlay = cv2.VideoWriter(
        str(Path(OUTPUT_DIR) / "overlay.avi"),
        fourcc, fps, (width, height)
    )
    
    # Store results
    results = []
    frame_num = 0
    all_masks = None
    
    print("\nProcessing video frames...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Run Cellpose segmentation
        try:
            result = model.eval(
                gray,
                diameter=DIAMETER,
                channels=[0, 0]
            )
            masks = result[0]
            flows = result[1]
            styles = result[2]
        except Exception as e:
            print(f"Error segmenting frame {frame_num}: {e}")
            masks = np.zeros_like(gray)
        
        # Classify cells and calculate metrics
        cell_data = {
            'frame': frame_num,
            'circular_count': 0,
            'fixed_count': 0,
            'dead_count': 0,
            'fragment_count': 0,
            'circular_area': [],
            'fixed_area': [],
            'dead_area': [],
            'fragment_area': [],
            'avg_brightness': float(np.mean(gray))
        }
        
        # Get properties for each cell
        unique_cells = np.unique(masks)[1:]  # Exclude background (0)
        
        for cell_id in unique_cells:
            cell_mask = masks == cell_id
            area = np.sum(cell_mask)
            
            # Get brightness within cell
            cell_brightness = np.mean(gray[cell_mask])
            
            # Classify cell
            class_name, class_id = classify_cell(frame_num, area, cell_brightness)
            
            # Update counts
            if class_name == 'circular':
                cell_data['circular_count'] += 1
                cell_data['circular_area'].append(area)
            elif class_name == 'fixed':
                cell_data['fixed_count'] += 1
                cell_data['fixed_area'].append(area)
            elif class_name == 'dead':
                cell_data['dead_count'] += 1
                cell_data['dead_area'].append(area)
            elif class_name == 'fragment':
                cell_data['fragment_count'] += 1
                cell_data['fragment_area'].append(area)
        
        # Calculate averages
        for key in ['circular_area', 'fixed_area', 'dead_area', 'fragment_area']:
            if cell_data[key]:
                cell_data[f"avg_{key}"] = float(np.mean(cell_data[key]))
            else:
                cell_data[f"avg_{key}"] = 0.0
            del cell_data[key]
        
        results.append(cell_data)
        
        # Create mask visualizations
        mask_display = np.zeros((height, width, 3), dtype=np.uint8)
        for cell_id in unique_cells:
            cell_mask = masks == cell_id
            area = np.sum(cell_mask)
            _, class_id = classify_cell(frame_num, area, 0)
            
            colors = [
                (0, 255, 0),      # Green - Circular
                (255, 255, 0),    # Cyan - Fixed
                (0, 0, 255),      # Red - Dead
                (128, 128, 128)   # Gray - Fragments
            ]
            mask_display[cell_mask] = colors[class_id]
        
        # Create overlay (original + mask)
        overlay = cv2.addWeighted(frame, 0.7, mask_display, 0.3, 0)
        
        # Write frames
        out_masks.write(mask_display)
        out_overlay.write(overlay)
        
        frame_num += 1
        if frame_num % 10 == 0:
            print(f"  Processed {frame_num}/{frame_count} frames")
    
    # Release video writers
    cap.release()
    out_masks.release()
    out_overlay.release()
    
    return results

def save_results(results):
    """Save results to CSV"""
    df = pd.DataFrame(results)
    csv_path = Path(OUTPUT_DIR) / "cell_metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Results saved to: {csv_path}")
    print("\nFirst 5 rows:")
    print(df.head())
    return df

def main():
    """Main execution"""
    print("=" * 60)
    print("Cell Segmentation Task - Cellpose")
    print("=" * 60)
    
    # Check input file
    if not Path(VIDEO_INPUT_PATH).exists():
        print(f"ERROR: Video file not found at {VIDEO_INPUT_PATH}")
        print("Please update VIDEO_INPUT_PATH in config.py")
        return
    
    # Load video
    print(f"\nLoading video: {VIDEO_INPUT_PATH}")
    cap, fps, width, height, frame_count = load_video(VIDEO_INPUT_PATH)
    print(f"  FPS: {fps}, Resolution: {width}x{height}, Frames: {frame_count}")
    
    # Initialize model
    model = initialize_model()
    
    # Process video
    results = process_video(model, cap, fps, width, height, frame_count)
    
    # Save results
    df = save_results(results)
    
    print("\n" + "=" * 60)
    print("✓ Segmentation complete!")
    print(f"✓ Output files saved to: {OUTPUT_DIR}")
    print("  - segmentation_masks.avi")
    print("  - overlay.avi")
    print("  - cell_metrics.csv")
    print("=" * 60)

if __name__ == "__main__":
    main()