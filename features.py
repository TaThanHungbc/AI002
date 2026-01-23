# features.py
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.morphology import skeletonize
from scipy.spatial.distance import cdist
from preprocess import process_image_for_model, adaptive_thresh, postprocess_to_rgb
from math import sqrt

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

def compute_ssim_groups(patches, img_gray, size=(64,64), thresh=0.98):
    """
    Compute SSIM pairwise across patches (resized to fixed size).
    Return list of groups with count >=2 and examples.
    """
    n = len(patches)
    if n < 2:
        return []
    imgs = []
    for p in patches:
        im = p['patch']
        # blend with grayscale to get texture: overlay mask area from original gray
        x,y,w,h = p['bbox']
        crop_gray = img_gray[y:y+h, x:x+w]
        # normalize and resize
        if crop_gray.size == 0:
            crop_gray = np.zeros((1,1), dtype=np.uint8)
        imr = cv2.resize(crop_gray, size, interpolation=cv2.INTER_LINEAR)
        imgs.append(imr)
    groups = []
    used = set()
    for i in range(n):
        if i in used: continue
        group = [i]
        for j in range(i+1, n):
            val = ssim(imgs[i], imgs[j])
            if val >= thresh:
                group.append(j)
        if len(group) >= 2:
            groups.append(group)
            used.update(group)
    # return groups with counts
    out = []
    for g in groups:
        # approximate line by y coordinate of first
        idx0 = g[0]
        y = patches[idx0]['centroid'][1]
        out.append({'indices': g, 'count': len(g), 'line_y': y})
    return out

def spacing_regularilty(patches):
    """
    Group patches by line (y coordinate clustering), compute kerning (x distance between consecutive boxes)
    Return CV across all distances and a flag if CV <= 0.03.
    """
    if len(patches) < 2:
        return {'cv': None, 'flag': False}
    # sort by y, cluster into lines using simple threshold = median height
    heights = [p['bbox'][3] for p in patches]
    if len(heights)==0:
        return {'cv': None, 'flag': False}
    median_h = np.median(heights)
    # assign to lines
    lines = {}
    for p in patches:
        cx, cy = p['centroid']
        assigned = False
        for ky in list(lines.keys()):
            if abs(cy - ky) < median_h: # same line
                lines[ky].append(p)
                assigned = True
                break
        if not assigned:
            lines[cy] = [p]
    distances = []
    for ky, arr in lines.items():
        arr_sorted = sorted(arr, key=lambda q: q['bbox'][0])  # by x
        for i in range(len(arr_sorted)-1):
            x1 = arr_sorted[i]['bbox'][0] + arr_sorted[i]['bbox'][2]/2
            x2 = arr_sorted[i+1]['bbox'][0] + arr_sorted[i+1]['bbox'][2]/2
            distances.append(abs(x2-x1))
    if len(distances) == 0:
        return {'cv': None, 'flag': False}
    mu = np.mean(distances)
    sigma = np.std(distances)
    cv = sigma / mu if mu>0 else 0.0
    flag = cv <= 0.03
    return {'cv': float(cv), 'flag': flag, 'mean_distance': float(mu), 'std': float(sigma), 'n_distances': len(distances)}

def stroke_variability_and_pressure(mask, img_gray):
    """
    Estimate stroke width using distance transform on mask.
    Compute CV_w, std_I, R (edge roughness normalized).
    """
    binmask = (mask>0).astype(np.uint8)
    # distance transform on foreground (distance to background) gives half stroke width approx
    dist = cv2.distanceTransform(255 - binmask*255, cv2.DIST_L2, 5)  # distance on background => not directly
    # better: invert mask, distance to background from mask skeleton? alternative compute thickness via medial axis
    # We'll use: distance transform on inverse mask to get background distances then approximate widths on mask pixels
    dt = cv2.distanceTransform(binmask*255, cv2.DIST_L2, 5)
    stroke_pixels = dt[dt>0]
    if stroke_pixels.size == 0:
        mean_w = 0.0
        cv_w = 0.0
    else:
        # approximate stroke width = 2 * distance transform
        widths = 2.0 * stroke_pixels
        mean_w = float(np.mean(widths))
        std_w = float(np.std(widths))
        cv_w = std_w / mean_w if mean_w>0 else 0.0

    # std_I: intensity std of ink pixels in grayscale normalized [0,1]
    ink_pixels = img_gray[binmask>0].astype(np.float32)/255.0
    std_I = float(np.std(ink_pixels)) if ink_pixels.size>0 else 0.0

    # Edge roughness R: find contours and compute mean deviation of contour radius to centroid normalized by mean stroke width
    contours, _ = cv2.findContours((binmask*255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    deviations = []
    for cnt in contours:
        if len(cnt) < 5:
            continue
        M = cv2.moments(cnt)
        if M['m00'] == 0:
            continue
        cx = M['m10']/M['m00']
        cy = M['m01']/M['m00']
        dists = np.sqrt(((cnt[:,0,0]-cx)**2 + (cnt[:,0,1]-cy)**2))
        mean_r = np.mean(dists)
        dev = np.mean(np.abs(dists - mean_r))
        deviations.append(dev)
    mean_dev = float(np.mean(deviations)) if len(deviations)>0 else 0.0
    R = (mean_dev / mean_w) if mean_w>0 else 0.0

    return {'cv_w': float(cv_w), 'mean_w': float(mean_w), 'std_I': float(std_I), 'R': float(R)}

def baseline_alignment(patches):
    """
    For each line estimate baseline as median of bottom y of bboxes, compute std of bottoms relative to baseline.
    Return average std across lines.
    """
    if len(patches) == 0:
        return {'std_baseline': None, 'flag': False}
    # group into lines similar to spacing
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
        bottoms = [p['bbox'][1] + p['bbox'][3] for p in arr]
        if len(bottoms) < 2:
            continue
        sigma = float(np.std(bottoms))
        stds.append(sigma)
    if len(stds)==0:
        return {'std_baseline': None, 'flag': False}
    mean_std = float(np.mean(stds))
    flag = mean_std <= 0.5
    return {'std_baseline': mean_std, 'flag': flag, 'n_lines': len(stds)}

def continuity_score(patches, mask):
    """
    For adjacent patches in same line, compute gradient continuity at junction.
    Return number of junctions with continuity < 0.4 and overall mean continuity.
    """
    if len(patches) < 2:
        return {'mean_cont': None, 'n_bad': 0, 'bad_examples': []}
    # group lines
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
            # compute junction box between right half of a and left half of b
            x1 = int(a['bbox'][0] + 0.6*a['bbox'][2])
            x2 = int(b['bbox'][0] + 0.4*b['bbox'][2])
            y1 = int(min(a['bbox'][1], b['bbox'][1]) - 2)
            y2 = int(max(a['bbox'][1]+a['bbox'][3], b['bbox'][1]+b['bbox'][3]) + 2)
            if x2 <= x1 or y2<=y1:
                # tiny or overlapping; crop small region around seam
                x1 = max(0, a['bbox'][0]+a['bbox'][2]-5)
                x2 = min(mask.shape[1]-1, b['bbox'][0]+5)
                y1 = max(0, int(min(a['bbox'][1], b['bbox'][1])))
                y2 = min(mask.shape[0]-1, int(max(a['bbox'][1]+a['bbox'][3], b['bbox'][1]+b['bbox'][3])))
            # guard bounds
            x1 = max(0, x1); x2 = min(mask.shape[1]-1, x2)
            y1 = max(0, y1); y2 = min(mask.shape[0]-1, y2)
            if x2<=x1 or y2<=y1:
                continue
            region_grad = grad_mag[y1:y2, x1:x2]
            # measure continuity as normalized inverse of gradient magnitude variance:
            # if gradient is low and smooth -> continuity high (close to 1)
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

def analyze_image_for_evidence(image_path):
    """
    High-level function:
    - produce binary mask + grayscale (using preprocess pipeline)
    - extract patches
    - compute all 5 checks and return structured result
    """
    # reuse preprocess: get grayscale and mask
    # process_image_for_model returns rgb and laplacian; we need gray and mask
    import cv2
    img_bgr = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Cannot read image")
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # CLAHE and deskew from preprocess
    from preprocess import apply_clahe_gray, deskew_image, adaptive_thresh
    clahe = apply_clahe_gray(gray)
    deskewed = deskew_image(clahe)
    mask = adaptive_thresh(deskewed)  # binary inverted (text=white)
    # convert mask to 0/1
    mask01 = (mask>0).astype(np.uint8)
    patches = extract_patches_from_mask(mask)
    # SSIM groups
    ssim_groups = compute_ssim_groups(patches, deskewed, size=(64,64), thresh=0.98)
    ssim_flag = len(ssim_groups) > 0
    # Spacing
    spacing = spacing_regularilty(patches)
    spacing_flag = spacing.get('flag', False)
    # Stroke variability
    stroke = stroke_variability_and_pressure(mask01, deskewed)
    stroke_flag = (stroke['cv_w'] <= 0.03) and (stroke['std_I'] <= 0.06) and (stroke['R'] <= 0.12)
    # Baseline
    baseline = baseline_alignment(patches)
    baseline_flag = baseline.get('flag', False)
    # Continuity
    continuity = continuity_score(patches, mask01)
    continuity_flag = continuity.get('n_bad', 0) > 0

    result = {
        'ssim': {'flag': ssim_flag, 'groups': ssim_groups},
        'spacing': {'flag': spacing_flag, **spacing},
        'stroke': {'flag': stroke_flag, **stroke},
        'baseline': {'flag': baseline_flag, **baseline},
        'continuity': {'flag': continuity_flag, **continuity}
    }
    return result
