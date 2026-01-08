import os
import sys
import time
import torch
import cv2
import numpy as np
import argparse

# Add project root to sys.path
prj_path = os.path.dirname(os.path.abspath(__file__))
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.tracker.mambanut import MambaNUT
from lib.test.parameter.mambanut import parameters
from lib.test.evaluation.environment import env_settings

def run_speed_test(yaml_name, checkpoint_epoch=None):
    # 1. Get parameters
    params = parameters(yaml_name)
    
    # Override checkpoint if needed (optional)
    if checkpoint_epoch is not None:
        save_dir = env_settings().save_dir
        params.checkpoint = os.path.join(save_dir, "checkpoints/train/mambanut/%s/MambaNUT_ep%04d.pth.tar" %
                                     (yaml_name, checkpoint_epoch))
    
    # Set missing parameters
    params.debug = False
    
    print(f"Loading checkpoint: {params.checkpoint}")
    if not os.path.exists(params.checkpoint):
        print("Error: Checkpoint file not found!")
        return

    # 2. Build Tracker
    # Dataset name is just for logging, doesn't affect speed test logic
    tracker = MambaNUT(params, "test_dataset")

    # 3. Prepare Dummy Data
    # Assuming standard video resolution, e.g., 720p or 1080p
    # The tracker resizes internally, so input size matters slightly for pre-processing but not for model inference
    H, W = 720, 1280
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    
    # Random bbox [x, y, w, h]
    init_bbox = [W//4, H//4, W//4, H//4]
    info = {'init_bbox': init_bbox}

    # 4. Initialize (Warm up)
    print("Initializing tracker...")
    tracker.initialize(image, info)

    # 5. Warm up loop
    T_w = 50
    print(f"Warming up for {T_w} frames...")
    for _ in range(T_w):
        tracker.track(image)

    # 6. Speed Test
    T_t = 500
    print(f"Testing speed for {T_t} frames...")
    
    # Use CUDA events for accurate timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(T_t):
        tracker.track(image)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event) # in milliseconds
    elapsed_time_s = elapsed_time_ms / 1000.0
    
    avg_lat = elapsed_time_s / T_t
    fps = T_t / elapsed_time_s
    
    print("\n" + "="*30)
    print(f"Results for {yaml_name}")
    print(f"Total time: {elapsed_time_s:.4f} s for {T_t} frames")
    print(f"Average Latency: {avg_lat*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Speed Test for MambaNUT')
    parser.add_argument('--config', type=str, default='mambar_small_patch16_224', help='yaml configure file name')
    parser.add_argument('--epoch', type=int, default=300, help='epoch number of checkpoint')
    
    args = parser.parse_args()
    
    # Set device
    torch.cuda.set_device(0)
    
    # Mock parameters function expects env settings
    # We rely on lib.test.evaluation.environment to find paths via local.py
    
    run_speed_test(args.config, args.epoch)
