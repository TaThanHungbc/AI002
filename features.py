# features.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from preprocess import apply_clahe_gray, deskew_image, adaptive_thresh

# ----------------------------
# Helpers: patch extraction
# ----------------------------
def extract_patches_from_mask(mask, min_area=50, max_area=5000):
    """
    mask: binary uint8 (255 foreground)
    return: list of dicts {bbox:(x,y,w,h), patch (uint8 image mask), centroid (cx,cy)}
    """
    mask_bin = (mask > 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    patches = []
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < min_area or area > max_area:
            continue
        comp_mask = (labels == i).astype(np.uint8) * 255
        patch = comp_mask[y:y+h, x:x+w]
        cx, cy = centroids[i]
        patches.append({'bbox': (x,y,w,h), 'patch': patch, 'centroid': (cx,cy), 'area': area})
    return patches

# ----------------------------
# SSIM: count similar pairs
# ----------------------------
def compute_shape_similarity(patches, img_gray, size=(64,64), hu_thresh=0.003, orb_match_thresh=0.25, max_checks=2000):
    """
    Kết hợp Hu moments + ORB matching để đếm số cặp 'rất giống nhau'.
    - patches: list như trước (bbox, patch)
    - img_gray: grayscale của ảnh gốc (không resize)
    Trả về {'n_pairs': int, 'flag': bool}
    """
    n = len(patches)
    if n < 2:
        return {'n_pairs': 0, 'flag': False}

    # prepare crops at original size (not resized to tiny) to keep detail
    crops = []
    for p in patches:
        x,y,w,h = p['bbox']
        crop = img_gray[y:y+h, x:x+w]
        if crop.size == 0:
            crop = np.zeros((1,1), dtype=np.uint8)
        crops.append(crop)

    # precompute Hu moments
    hu_list = []
    for im in crops:
        # binarize for moment calc
        _, th = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        moments = cv2.moments(th)
        hu = cv2.HuMoments(moments).flatten()
        # log scale to stabilize
        hu_log = -np.sign(hu) * np.log10(np.abs(hu)+1e-12)
        hu_list.append(hu_log)

    # ORB descriptors (for fallback)
    orb = cv2.ORB_create(nfeatures=200)
    orb_desc = []
    for im in crops:
        # try to detect on resized patch
        im_res = cv2.resize(im, (128,128), interpolation=cv2.INTER_LINEAR)
        kp, des = orb.detectAndCompute(im_res, None)
        orb_desc.append(des)

    count = 0
    checked = 0
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    for i in range(n):
        for j in range(i+1, n):
            checked += 1
            if checked > max_checks:
                break
            # first check Hu distance (fast)
            hu_dist = np.linalg.norm(hu_list[i] - hu_list[j])
            if hu_dist <= hu_thresh:
                count += 1
                continue
            # fallback to ORB matching if Hu inconclusive and descriptors exist
            des1 = orb_desc[i]
            des2 = orb_desc[j]
            if des1 is not None and des2 is not None:
                matches = bf.match(des1, des2)
                if len(matches) == 0:
                    continue
                # ratio of good matches by number of keypoints
                good = sum(1 for m in matches if m.distance < 60)
                denom = max(len(des1), len(des2))
                score = good / denom if denom>0 else 0.0
                if score >= orb_match_thresh:
                    count += 1
    flag = count > 0
    return {'n_pairs': int(count), 'flag': flag}

# ----------------------------
# Spacing regularity
# ----------------------------
def spacing_regularilty(patches):
    if len(patches) < 2:
        return {'cv': None, 'flag': False}
    # cluster by y using k-means-like (but simple agglomerative)
    heights = [p['bbox'][3] for p in patches]
    median_h = np.median(heights) if len(heights)>0 else 10
    # group by line as before
    lines = {}
    for p in patches:
        cx, cy = p['centroid']
        assigned = False
        for ky in list(lines.keys()):
            if abs(cy - ky) < median_h:
                lines[ky].append(p)
                assigned = True
                break
        if not assigned:
            lines[cy] = [p]
    cvs = []
    n_distances = 0
    distances_all = []
    for ky, arr in lines.items():
        arr_sorted = sorted(arr, key=lambda q: q['bbox'][0])
        dists = []
        for i in range(len(arr_sorted)-1):
            x1 = arr_sorted[i]['bbox'][0] + arr_sorted[i]['bbox'][2]/2
            x2 = arr_sorted[i+1]['bbox'][0] + arr_sorted[i+1]['bbox'][2]/2
            d = abs(x2-x1)
            dists.append(d)
            distances_all.append(d)
        if len(dists) >= 2:
            mu = np.mean(dists)
            sigma = np.std(dists)
            cvs.append(sigma / mu if mu>0 else 0.0)
            n_distances += len(dists)
    if len(distances_all) == 0:
        return {'cv': None, 'flag': False}
    global_mu = np.mean(distances_all)
    global_sigma = np.std(distances_all)
    global_cv = global_sigma / global_mu if global_mu>0 else 0.0
    flag = global_cv <= 0.03
    return {'cv': float(global_cv), 'flag': flag, 'mean_distance': float(global_mu), 'std': float(global_sigma), 'n_distances': n_distances}

# ----------------------------
# Stroke variability & pressure proxy
# ----------------------------

from skimage.morphology import skeletonize
def stroke_variability_and_pressure(mask, img_gray):
    binmask = (mask>0).astype(np.uint8)
    if binmask.sum() == 0:
        return {'cv_w':0.0, 'mean_w':0.0, 'std_I':0.0, 'R':0.0}
    # distance transform of mask (distance to background)
    dt = cv2.distanceTransform((binmask*255).astype(np.uint8), cv2.DIST_L2, 5)
    # compute skeleton in binary, use skimage's skeletonize on boolean
    skel = skeletonize(binmask.astype(bool)).astype(np.uint8)
    # sample DT values at skeleton points -> local half-width
    skel_vals = dt[skel==1]
    if skel_vals.size == 0:
        mean_w = 0.0
        cv_w = 0.0
    else:
        widths = 2.0 * skel_vals  # approximate stroke widths
        mean_w = float(np.mean(widths))
        std_w = float(np.std(widths))
        cv_w = std_w / mean_w if mean_w>0 else 0.0

    # intensity std of ink pixels in original gray (normalized)
    ink_vals = img_gray[binmask>0].astype(np.float32)/255.0
    std_I = float(np.std(ink_vals)) if ink_vals.size>0 else 0.0

    # edge roughness R: use contour curvature variance normalized by mean width
    contours, _ = cv2.findContours((binmask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    deviations = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        # compute curvature-like deviation: distance from contour points to fitted ellipse/mean radius
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10']/M['m00']; cy = M['m01']/M['m00']
        dists = np.sqrt(((cnt[:,0,0]-cx)**2 + (cnt[:,0,1]-cy)**2))
        mean_r = np.mean(dists)
        dev = np.mean(np.abs(dists - mean_r))
        deviations.append(dev)
    mean_dev = float(np.mean(deviations)) if len(deviations)>0 else 0.0
    R = (mean_dev / mean_w) if mean_w>0 else 0.0

    return {'cv_w': float(cv_w), 'mean_w': float(mean_w), 'std_I': float(std_I), 'R': float(R)}

# ----------------------------
# Baseline alignment
# ----------------------------

from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import LinearRegression

def baseline_alignment(patches):
    if len(patches) == 0:
        return {'std_baseline': None, 'flag': False}
    heights = [p['bbox'][3] for p in patches]
    median_h = np.median(heights) if len(heights)>0 else 10
    lines = {}
    for p in patches:
        cx, cy = p['centroid']
        assigned = False
        for ky in list(lines.keys()):
            if abs(cy - ky) < median_h:
                lines[ky].append(p)
                assigned = True
                break
        if not assigned:
            lines[cy] = [p]
    stds = []
    for ky, arr in lines.items():
        if len(arr) < 3:
            continue
        X = np.array([ [p['centroid'][0]] for p in arr ])  # x positions
        Y = np.array([ p['bbox'][1] + p['bbox'][3] for p in arr ])  # bottom y
        try:
            ransac = RANSACRegressor(LinearRegression(), residual_threshold=2.0, random_state=42)
            ransac.fit(X, Y)
            inlier_mask = ransac.inlier_mask_
            residuals = np.abs(Y[inlier_mask] - ransac.predict(X[inlier_mask]))
            if residuals.size > 0:
                stds.append(float(np.std(residuals)))
        except Exception:
            # fallback to simple std
            stds.append(float(np.std(Y)))
    if len(stds) == 0:
        return {'std_baseline': None, 'flag': False}
    mean_std = float(np.mean(stds))
    flag = mean_std <= 0.5
    return {'std_baseline': mean_std, 'flag': flag, 'n_lines': len(stds)}

# ----------------------------
# Continuity score
# ----------------------------
def continuity_score(patches, mask):
    if len(patches) < 2:
        return {'mean_cont': None, 'n_bad': 0, 'bad_examples': []}
    heights = [p['bbox'][3] for p in patches] if len(patches)>0 else [10]
    median_h = np.median(heights)
    lines = {}
    for p in patches:
        cx, cy = p['centroid']
        assigned = False
        for ky in list(lines.keys()):
            if abs(cy - ky) < median_h:
                lines[ky].append(p)
                assigned = True
                break
        if not assigned:
            lines[cy] = [p]
    cont_values = []
    bad_count = 0
    bad_examples = []
    edges = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
    edges_y = cv2.Sobel(mask.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(edges**2 + edges_y**2) + 1e-8
    for ky, arr in lines.items():
        arr_sorted = sorted(arr, key=lambda q: q['bbox'][0])
        for i in range(len(arr_sorted)-1):
            a = arr_sorted[i]
            b = arr_sorted[i+1]
            x1 = int(a['bbox'][0] + 0.6*a['bbox'][2])
            x2 = int(b['bbox'][0] + 0.4*b['bbox'][2])
            y1 = int(min(a['bbox'][1], b['bbox'][1]) - 2)
            y2 = int(max(a['bbox'][1]+a['bbox'][3], b['bbox'][1]+b['bbox'][3]) + 2)
            if x2 <= x1 or y2<=y1:
                x1 = max(0, a['bbox'][0]+a['bbox'][2]-5)
                x2 = min(mask.shape[1]-1, b['bbox'][0]+5)
                y1 = max(0, int(min(a['bbox'][1], b['bbox'][1])))
                y2 = min(mask.shape[0]-1, int(max(a['bbox'][1]+a['bbox'][3], b['bbox'][1]+b['bbox'][3])))
            x1 = max(0, x1); x2 = min(mask.shape[1]-1, x2)
            y1 = max(0, y1); y2 = min(mask.shape[0]-1, y2)
            if x2<=x1 or y2<=y1:
                continue
            region_grad = grad_mag[y1:y2, x1:x2]
            mean_g = np.mean(region_grad)
            std_g = np.std(region_grad)
            cont = 1.0 - (std_g / (mean_g + 1e-8))
            cont = float(np.clip(cont, 0.0, 1.0))
            cont_values.append(cont)
            if cont < 0.4:
                bad_count += 1
                bad_examples.append({'bbox_a': a['bbox'], 'bbox_b': b['bbox'], 'cont': cont})
    mean_cont = float(np.mean(cont_values)) if len(cont_values)>0 else None
    return {'mean_continuity': mean_cont, 'n_bad': bad_count, 'bad_examples': bad_examples}

# ----------------------------
# High-level analyzer
# ----------------------------
def analyze_image_for_evidence(image_path):
    """
    Return a dict with keys: ssim, spacing, stroke, baseline, continuity
    Each value contains {'flag': bool, ...stats...}
    """
    import cv2
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot read image")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    clahe = apply_clahe_gray(gray)
    deskewed = deskew_image(clahe)
    mask = adaptive_thresh(deskewed)  # binary inverted (text=white)
    mask01 = (mask>0).astype(np.uint8)
    patches = extract_patches_from_mask(mask)
    ssim_res = compute_shape_similarity(patches, deskewed, size=(64,64),
                                        hu_thresh=0.003, orb_match_thresh=0.25, max_checks=2000)
    spacing = spacing_regularilty(patches)
    stroke = stroke_variability_and_pressure(mask01, deskewed)
    stroke_flag = (stroke['cv_w'] <= 0.03) and (stroke['std_I'] <= 0.06) and (stroke['R'] <= 0.12)
    baseline = baseline_alignment(patches)
    continuity = continuity_score(patches, mask01)

    result = {
        'ssim': {'flag': ssim_res['flag'], 'n_pairs': ssim_res['n_pairs']},
        'spacing': {'flag': spacing.get('flag', False), **spacing},
        'stroke': {'flag': stroke_flag, **stroke},
        'baseline': {'flag': baseline.get('flag', False), **baseline},
        'continuity': {'flag': continuity.get('n_bad', 0) > 0, **continuity}
    }
    return result
