# --- START OF FILE preprocess.py ---
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from pphelpers import *

# --- CẤU HÌNH ---
RAW_DIR = 'datasets_raw'
DATA_DIR = 'datasets'
CATEGORIES = ['AI', 'Human']
IMG_SIZE = (128, 128)

# ==========================================
TEST1FILE = False    # Đặt True để test crop ngon chưa
TEST_FILENAME = 'vh.jpg' # File bạn đang test
# ==========================================

def process_pipeline(img, debug_prefix=None):
    if img is None: raise ValueError("Empty Image")

    # 1. Xoay
    img = correct_orientation(img)

    # 2. AI Clean (Nền đen - Chữ trắng)
    clean_binary = ai_process_debug(img, debug_prefix)

    # [BƯỚC MỚI] 3. Auto Crop (Cắt sát chữ)
    cropped_img = auto_crop_content(clean_binary)
    if debug_prefix: save_debug(cropped_img, debug_prefix, "06b_cropped_content")

    # 4. Resize & Padding (Phóng to ảnh crop lên 128x128)
    final_img = pad_and_resize(cropped_img, IMG_SIZE)
    if debug_prefix: save_debug(final_img, debug_prefix, "07_final_result")

    return final_img

def preprocess_dataset():
    print(f"--- STARTING ({'TEST MODE' if TEST1FILE else 'BATCH MODE'}) ---")
    
    processed_count = 0
    target_cats = ['Human'] if TEST1FILE else CATEGORIES

    for category in target_cats: 
        src_dir = os.path.join(RAW_DIR, category)
        dst_dir = os.path.join(DATA_DIR, category)
        
        if not os.path.exists(dst_dir): os.makedirs(dst_dir)
        if not os.path.exists(src_dir): continue
            
        files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if TEST1FILE:
            if TEST_FILENAME:
                files = [TEST_FILENAME] if TEST_FILENAME in files else []
                if not files: print(f"File {TEST_FILENAME} not found!"); return
            else:
                files = files[:1]
            print(f"Processing ONLY file: {files[0]}")

        for filename in tqdm(files, desc=category):
            try:
                src_path = os.path.join(src_dir, filename)
                dst_path = os.path.join(dst_dir, filename)
                
                img = cv2.imread(src_path)
                
                debug_prefix = None
                if TEST1FILE:
                    debug_prefix = os.path.join(dst_dir, f"DEBUG_{filename.split('.')[0]}")
                    print(f"\nSaving debug images to: {dst_dir}")

                final_img = process_pipeline(img, debug_prefix=debug_prefix)
                cv2.imwrite(dst_path, final_img)
                processed_count += 1
                
                if TEST1FILE:
                    print("\n--- TEST MODE FINISHED ---")
                    return 

            except Exception as e:
                print(f"[ERR] {filename}: {e}")
                import traceback
                traceback.print_exc()

def save_to_numpy():
    if TEST1FILE: return
    print("--- SAVING TO NPY ---")
    X, y = [], []
    for label, category in enumerate(CATEGORIES):
        folder = os.path.join(DATA_DIR, category)
        if not os.path.exists(folder): continue
        for filename in tqdm(os.listdir(folder)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')) and "DEBUG" not in filename:
                try:
                    img = Image.open(os.path.join(folder, filename)).convert('L')
                    img = img.resize(IMG_SIZE)
                    arr = np.array(img).astype('float32') / 255.0
                    X.append(arr)
                    y.append(label)
                except: pass
    if X:
        X = np.array(X).reshape(-1, IMG_SIZE[0], IMG_SIZE[1], 1)
        np.save("X_data.npy", X)
        np.save("y_data.npy", np.array(y))
        print("Saved NPY.")

def preprocess_single_image(path_or_input):
    img = None
    if isinstance(path_or_input, str): img = cv2.imread(path_or_input)
    elif isinstance(path_or_input, np.ndarray): img = path_or_input
    elif isinstance(path_or_input, Image.Image): 
        img = cv2.cvtColor(np.array(path_or_input), cv2.COLOR_RGB2BGR)
    if img is None: raise ValueError("Invalid Input")

    processed = process_pipeline(img, debug_prefix=None)
    norm = processed.astype('float32') / 255.0
    return norm.reshape(1, IMG_SIZE[0], IMG_SIZE[1], 1)

if __name__ == "__main__":
    preprocess_dataset()
    save_to_numpy()