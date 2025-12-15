import os
import time
import random
import cv2
import numpy as np
from faker import Faker
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from tqdm import tqdm

# --- CẤU HÌNH NGƯỜI DÙNG ---
OUTPUT_DIR = os.path.join('datasets_raw', 'AI')
NUM_SAMPLES = 50           # Số lượng ảnh
TIMEOUT = 15               
TARGET_W = 1200             # Chiều rộng ảnh cắt ra
TARGET_H = 500             # Chiều cao ảnh cắt ra

def init_driver():
    """Khởi tạo Chrome"""
    chrome_options = Options()
    # chrome_options.add_argument("--headless") # Nên để hiện để check
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--window-size=1280,800")
    chrome_options.add_argument("--no-sandbox")
    
    driver = webdriver.Chrome(options=chrome_options)
    return driver

def apply_random_settings(driver):
    """Speed Max, Legibility & Width Random"""
    try:
        sliders = driver.find_elements(By.CSS_SELECTOR, "input[type='range']")
        if len(sliders) >= 3:
            # 1. Speed -> Max
            driver.execute_script("arguments[0].value = arguments[0].max;", sliders[0])
            driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", sliders[0])
            
            # 2. Legibility -> Random
            rand_leg = random.uniform(float(sliders[1].get_attribute("min")), float(sliders[1].get_attribute("max")))
            driver.execute_script(f"arguments[0].value = {rand_leg};", sliders[1])
            driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", sliders[1])
            
            # 3. Stroke Width -> Random
            rand_stroke = random.uniform(float(sliders[2].get_attribute("min")), float(sliders[2].get_attribute("max")))
            driver.execute_script(f"arguments[0].value = {rand_stroke};", sliders[2])
            driver.execute_script("arguments[0].dispatchEvent(new Event('input', { bubbles: true }));", sliders[2])
    except: pass

def crop_center_fixed_size(image_path, target_w, target_h):
    """Cắt ảnh từ tâm"""
    img = cv2.imread(image_path)
    if img is None: return False
    
    h, w = img.shape[:2]
    canvas = np.ones((target_h, target_w, 3), dtype=np.uint8) * 255
    
    center_x, center_y = w // 2, h // 2
    
    crop_x1 = center_x - target_w // 2
    crop_y1 = center_y - target_h // 2
    crop_x2 = center_x + target_w // 2
    crop_y2 = center_y + target_h // 2
    
    paste_x1, paste_y1 = 0, 0
    paste_x2, paste_y2 = target_w, target_h
    
    if crop_x1 < 0: paste_x1 = -crop_x1; crop_x1 = 0
    if crop_y1 < 0: paste_y1 = -crop_y1; crop_y1 = 0
    if crop_x2 > w: paste_x2 = target_w - (crop_x2 - w); crop_x2 = w
    if crop_y2 > h: paste_y2 = target_h - (crop_y2 - h); crop_y2 = h
        
    region = img[crop_y1:crop_y2, crop_x1:crop_x2]
    canvas[paste_y1:paste_y2, paste_x1:paste_x2] = region
    
    cv2.imwrite(image_path, canvas)
    return True

def generate_calligrapher_fixed():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    
    print("--- KHỞI ĐỘNG CHROME ---")
    driver = init_driver()
    wait = WebDriverWait(driver, TIMEOUT)
    fake = Faker()
    
    try:
        driver.get("https://www.calligrapher.ai/")
        time.sleep(3) 
        
        print(f"--- BẮT ĐẦU CÀO {NUM_SAMPLES} ẢNH ---")
        
        for i in tqdm(range(NUM_SAMPLES)):
            try:
                # 1. Random Text
                text = fake.word()
                if random.random() > 0.5: text = text.title()
                
                # 2. Tìm input
                input_box = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type='text']")))
                
                # 3. Xóa cũ & Nhập mới
                input_box.send_keys(Keys.CONTROL + "a")
                input_box.send_keys(Keys.DELETE)
                input_box.send_keys(text)
                
                # 4. Settings
                apply_random_settings(driver)
                
                # 5. Bấm ENTER để vẽ
                input_box.send_keys(Keys.ENTER)
                
                # 6. Chờ vẽ
                time.sleep(0.5)
                # Đợi cho có path xuất hiện
                try:
                    wait.until(lambda d: len(d.find_elements(By.CSS_SELECTOR, "svg path")) > 0)
                except:
                    driver.refresh(); time.sleep(2); continue

                time.sleep(0.5) 
                
                # 7. TÌM ĐÚNG THẺ SVG (FIX LỖI CHỤP NHẦM ICON)
                svgs = driver.find_elements(By.TAG_NAME, "svg")
                target_svg = None
                
                # Lọc qua các thẻ SVG, lấy cái nào to nhất (hoặc rộng > 500px)
                for svg in svgs:
                    # Lấy kích thước
                    size = svg.size
                    # Icon download bé tí (~24px), Canvas vẽ thì rất to (~1000px)
                    if size['width'] > 300: 
                        target_svg = svg
                        break
                
                if target_svg is None:
                    print("Không tìm thấy Canvas vẽ -> Skip")
                    continue

                # 8. Chụp ảnh đúng thẻ SVG to
                filename = f"AI_Real_{i:04d}_{text}.png"
                save_path = os.path.join(OUTPUT_DIR, filename)
                target_svg.screenshot(save_path)
                
                # 9. Crop từ tâm
                crop_center_fixed_size(save_path, TARGET_W, TARGET_H)
                
            except Exception as e:
                print(f" [Err {i}] {e}")
                try: driver.refresh(); time.sleep(2)
                except: pass
                
    finally:
        driver.quit()
        print(f"DONE. Folder: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_calligrapher_fixed()