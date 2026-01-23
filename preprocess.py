# preprocess.py
import cv2
import numpy as np
from math import atan2, degrees
from PIL import Image

def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def apply_clahe_gray(gray):
    # CLAHE trên ảnh xám
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def deskew_image(img_gray):
    # Dùng Canny + HoughLinesP để ước lượng góc nghiêng (skew)
    edges = cv2.Canny(img_gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    angles = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0]:
            angle = degrees(atan2(y2 - y1, x2 - x1))
            # chúng ta quan tâm các đường gần ngang
            if abs(angle) < 45:
                angles.append(angle)
    if len(angles) == 0:
        return img_gray  # không có gì để deskew
    median_angle = np.median(angles)
    # xoay bù lại
    (h, w) = img_gray.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(img_gray, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def adaptive_thresh(img_gray):
    # Adaptive threshold để lấy mask nét mực
    blur = cv2.GaussianBlur(img_gray, (3,3), 0)
    th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, 25, 10)
    # nhỏ morph open để loại nhiễu
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def postprocess_to_rgb(binary_mask, original_gray):
    # Kết hợp mask và grayscale để tạo ảnh 3-channel phù hợp cho pretrained model
    # Chúng ta có 2 lựa chọn: 1) dùng binary mask replicated 3 channels
    #                        2) blend mask với gray để giữ texture
    # Mình sẽ blend: mask as alpha over gray background -> RGB
    mask = binary_mask // 255
    blended = (original_gray * (1 - 0.7*mask)).astype(np.uint8)
    rgb = cv2.cvtColor(blended, cv2.COLOR_GRAY2RGB)
    return rgb

def process_image_for_model(path, debug=False, min_laplacian_var=100):
    """ Đọc ảnh từ path -> preprocess -> trả về image RGB (uint8) kích thước tùy chỉnh """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không đọc được ảnh: {path}")
    # nếu ảnh có orientation khác, rotate tự động (sử dụng EXIF) — cv2 không đọc EXIF, dùng PIL fallback
    try:
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil_exif = pil_img.getexif()
        # không ép buộc, bỏ qua nếu không có
    except Exception:
        pass

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lap_var = variance_of_laplacian(gray)
    # nếu ảnh quá mờ, vẫn tiếp tục nhưng log (main.py có thể reject)
    if debug:
        print(f"[preprocess] Laplacian variance: {lap_var:.2f}")

    # CLAHE
    clahe = apply_clahe_gray(gray)

    # Deskew
    deskewed = deskew_image(clahe)

    # Adaptive threshold
    mask = adaptive_thresh(deskewed)

    # Postprocess -> RGB
    rgb = postprocess_to_rgb(mask, deskewed)

    return rgb, lap_var

def process_image_resize(path, target_size=(224,224), debug=False, min_laplacian_var=100):
    img_rgb, lap_var = process_image_for_model(path, debug=debug, min_laplacian_var=min_laplacian_var)
    # resize with aspect fill -> center crop
    h, w = img_rgb.shape[:2]
    th, tw = target_size
    # scale to cover
    scale = max(th/h, tw/w)
    nh, nw = int(h*scale), int(w*scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    # center crop
    startx = (nw - tw)//2
    starty = (nh - th)//2
    cropped = resized[starty:starty+th, startx:startx+tw]
    return cropped, lap_var
