"""
Sample Asset Generator for Predictive Pavement Management System
Generates:
1. High-frequency 100 Hz 3-Axis Accelerometer CSV logs (IIT Guwahati mobile bicycle/rickshaw runs)
2. Realistic road test images (Damaged asphalt, clean asphalt, and non-road spam test images)
3. Initial Granica-compliant Parquet dataset
"""

import os
import numpy as np
import pandas as pd
import cv2
from pipeline import AccelerometerSignalEngine, IITG_CAMPUS_ROUTES, generate_granica_hackathon_dataset

def generate_sample_assets():
    os.makedirs("sample_data", exist_ok=True)
    
    # 1. Generate Accelerometer CSV logs
    accel_engine = AccelerometerSignalEngine(sampling_rate_hz=100.0)
    
    # Route 1: Core V North Loop
    df_core5 = accel_engine.generate_synthetic_mobile_run(
        duration_sec=50,
        route_info=IITG_CAMPUS_ROUTES[0],
        pothole_spikes=5
    )
    core5_path = os.path.join("sample_data", "sample_accel_run_core5.csv")
    df_core5.to_csv(core5_path, index=False)
    print(f"Generated Accelerometer Log: {core5_path} ({len(df_core5)} samples @ 100 Hz)")
    
    # Route 2: Brahmaputra Hostel Avenue
    df_brahma = accel_engine.generate_synthetic_mobile_run(
        duration_sec=60,
        route_info=IITG_CAMPUS_ROUTES[1],
        pothole_spikes=7
    )
    brahma_path = os.path.join("sample_data", "sample_accel_run_brahmaputra.csv")
    df_brahma.to_csv(brahma_path, index=False)
    print(f"Generated Accelerometer Log: {brahma_path} ({len(df_brahma)} samples @ 100 Hz)")

    # 2. Generate Road Images for Testing
    # A) Damaged Asphalt with Potholes and Surface Cracks
    img_damaged = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Asphalt gray base with granular texture
    base_color = np.array([58, 62, 65], dtype=np.uint8) # BGR
    img_damaged[:] = base_color
    noise = np.random.normal(0, 18, (720, 1280, 3)).astype(np.int16)
    img_damaged = np.clip(img_damaged.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Road markings (yellow center divider dashed line)
    for y in range(40, 720, 140):
        cv2.line(img_damaged, (640, y), (640, y + 80), (30, 210, 240), 12)
        
    # Paint Pothole 1 (sunken dark depression with rough rim)
    cv2.ellipse(img_damaged, (460, 480), (95, 60), 15, 0, 360, (22, 24, 25), -1)
    cv2.ellipse(img_damaged, (460, 480), (105, 68), 15, 0, 360, (40, 44, 46), 6) # jagged rim
    # Crack branching out
    pts1 = np.array([[460, 540], [480, 590], [530, 620], [560, 660]], np.int32)
    cv2.polylines(img_damaged, [pts1], False, (18, 20, 22), 4)

    # Paint Pothole 2
    cv2.ellipse(img_damaged, (820, 390), (75, 45), -20, 0, 360, (20, 22, 24), -1)
    cv2.ellipse(img_damaged, (820, 390), (82, 50), -20, 0, 360, (45, 48, 50), 5)
    
    damaged_path = os.path.join("sample_data", "sample_road_pothole.jpg")
    cv2.imwrite(damaged_path, img_damaged)
    print(f"Generated Road Image (Damaged): {damaged_path}")

    # B) Clean newly paved asphalt
    img_clean = np.zeros((720, 1280, 3), dtype=np.uint8)
    img_clean[:] = np.array([48, 50, 52], dtype=np.uint8)
    clean_noise = np.random.normal(0, 10, (720, 1280, 3)).astype(np.int16)
    img_clean = np.clip(img_clean.astype(np.int16) + clean_noise, 0, 255).astype(np.uint8)
    # Bright crisp lane markings
    for y in range(40, 720, 140):
        cv2.line(img_clean, (640, y), (640, y + 80), (240, 240, 240), 10)
    clean_path = os.path.join("sample_data", "sample_clean_road.jpg")
    cv2.imwrite(clean_path, img_clean)
    print(f"Generated Road Image (Clean): {clean_path}")

    # C) Spam image (Non-road: indoor room / brightly colored document)
    img_spam = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Bright colorful gradient (rejectable by AI spam filter)
    for c in range(1280):
        img_spam[:, c, 0] = int(220 * (c / 1280)) # Blue
        img_spam[:, c, 1] = int(180 * (1 - c / 1280)) # Green
        img_spam[:, c, 2] = 230 # Red
    cv2.putText(img_spam, "INDOOR NOTICE - NOT A ROAD", (260, 360),
                cv2.FONT_HERSHEY_DUPLEX, 1.4, (255, 255, 255), 3)
    spam_path = os.path.join("sample_data", "sample_spam_indoor.jpg")
    cv2.imwrite(spam_path, img_spam)
    print(f"Generated Non-Road Spam Test Image: {spam_path}")

    # 3. Pre-generate Granica Parquet dataset
    df_parquet, meta = generate_granica_hackathon_dataset(
        n_records=120,
        output_filepath="road_pothole_dataset.parquet"
    )
    print(f"Generated Granica Parquet Dataset: {len(df_parquet)} records")

if __name__ == "__main__":
    generate_sample_assets()
