from skimage.filters import threshold_sauvola
from skimage import io, color, img_as_ubyte
from PIL import Image
import numpy as np

def quick_clean_image(image_path):
    # 1. Đọc ảnh và chuyển xám
    img = io.imread(image_path)
    if len(img.shape) == 3:
        img_gray = color.rgb2gray(img)
    else:
        img_gray = img

    # 2. MAGIC: Sauvola Thresholding
    # Tự động tính toán ngưỡng sáng tối cục bộ (thay vì toàn bộ ảnh như Otsu)
    # Cực hiệu quả với giấy có ô ly hoặc bóng đổ
    window_size = 25
    thresh_sauvola = threshold_sauvola(img_gray, window_size=window_size)
    
    # Tạo ảnh nhị phân: Chỗ nào sáng hơn ngưỡng -> Trắng (True), tối hơn -> Đen (False)
    binary_sauvola = img_gray > thresh_sauvola
    
    # 3. Chuyển về format ảnh thông thường (0-255)
    clean_img = img_as_ubyte(binary_sauvola)
    
    return clean_img

# Test thử
clean = quick_clean_image("test.jpg")
io.imsave("clean_sauvola.jpg", clean)