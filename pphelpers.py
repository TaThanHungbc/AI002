# --- START OF FILE pphelpers.py ---
import cv2
import numpy as np
from rembg import remove, new_session
from PIL import Image

# Giữ nguyên cấu hình cũ
THRESHOLD = 155
ai_session = new_session(model_name="u2net")

def save_debug(img, prefix, step_name):
    if prefix is None or img is None: return
    filename = f"{prefix}_{step_name}.jpg"
    cv2.imwrite(filename, img)
    print(f"   -> Saved debug step: {filename}")

def correct_orientation(img):
    h, w = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return img

def ai_process_debug(img_cv2, debug_prefix=None):
    # --- GIỮ NGUYÊN CODE CŨ ---
    save_debug(img_cv2, debug_prefix, "01_original")

    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    result_pil = remove(img_pil, session=ai_session)
    
    res_np = np.array(result_pil)
    res_bgr = cv2.cvtColor(res_np, cv2.COLOR_RGBA2BGRA)
    save_debug(res_bgr, debug_prefix, "02_ai_raw_output")

    alpha_channel = res_np[:, :, 3]
    _, alpha_binary = cv2.threshold(alpha_channel, THRESHOLD, 255, cv2.THRESH_BINARY)
    save_debug(alpha_binary, debug_prefix, "04_alpha_thresholded")

    bg = Image.new("RGB", result_pil.size, (255, 255, 255))
    mask_pil = Image.fromarray(alpha_binary)
    bg.paste(result_pil, mask=mask_pil)
    
    clean_gray = np.array(bg.convert('L'))
    
    _, binary_final = cv2.threshold(clean_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    save_debug(binary_final, debug_prefix, "06_final_binary")

    return binary_final

def auto_crop_content(img_binary):
    """
    HÀM MỚI: Tự động cắt sát vùng có chữ (loại bỏ phần đen thừa thãi).
    """
    # Tìm tất cả các điểm trắng (chữ)
    coords = cv2.findNonZero(img_binary)
    
    # Nếu ảnh đen sì không có chữ thì trả về nguyên gốc
    if coords is None:
        return img_binary
    
    # Tìm hình chữ nhật bao quanh (Bounding Rect)
    x, y, w, h = cv2.boundingRect(coords)
    
    # Thêm lề (margin) một chút cho chữ đỡ bị sát mép quá
    margin = 10 
    h_img, w_img = img_binary.shape
    
    # Tính toán tọa độ cắt (đảm bảo không lòi ra ngoài ảnh)
    x_start = max(0, x - margin)
    y_start = max(0, y - margin)
    x_end = min(w_img, x + w + margin)
    y_end = min(h_img, y + h + margin)
    
    # Cắt ảnh
    cropped = img_binary[y_start:y_end, x_start:x_end]
    
    return cropped

def pad_and_resize(img, target_size=(128, 128)):
    # --- GIỮ NGUYÊN CODE CŨ ---
    target_w, target_h = target_size
    h, w = img.shape
    scale = min(target_w / w, target_h / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((target_h, target_w), dtype=np.uint8)
    x_offset = (target_w - nw) // 2
    y_offset = (target_h - nh) // 2
    
    canvas[y_offset:y_offset+nh, x_offset:x_offset+nw] = resized
    return canvas