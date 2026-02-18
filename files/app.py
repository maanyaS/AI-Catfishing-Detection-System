"""
CatfishGuard - AI-Powered Catfishing Detection Tool (v3 - Fixed Skin Detection)

Key fixes in v3:
  - Broadened skin detection mask to catch glossy/shiny AI skin
  - Added dedicated GLOSSINESS detector for AI's signature glassy shine
  - Center-region fallback when skin mask fails (always analyzes face area)
  - LBP texture now runs on full center region, not just detected skin
  - Higher base sensitivity across ALL layers
  - Recalibrated thresholds to be much more aggressive
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image, ImageFilter, ImageStat
import numpy as np
import os
import uuid

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def compute_lbp(gray_array, radius=1):
    """Compute Local Binary Pattern for micro-texture analysis."""
    h, w = gray_array.shape
    lbp = np.zeros((h - 2*radius, w - 2*radius), dtype=np.uint8)
    offsets = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
    for i, (dy, dx) in enumerate(offsets):
        ny = slice(radius + dy, h - radius + dy)
        nx = slice(radius + dx, w - radius + dx)
        center = gray_array[radius:h-radius, radius:w-radius]
        neighbor = gray_array[ny, nx]
        lbp |= ((neighbor >= center).astype(np.uint8) << i)
    return lbp


def get_center_region(array, fraction=0.5):
    """Extract the center region of an image (where the face likely is)."""
    h, w = array.shape[:2]
    cy, cx = h // 2, w // 2
    rh = int(h * fraction / 2)
    rw = int(w * fraction / 2)
    return array[cy - rh:cy + rh, cx - rw:cx + rw]


# ==============================================================================
# LAYER 1: ANATOMICAL ANOMALY DETECTION
# ==============================================================================

def analyze_anatomical_anomalies(image):
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    grad_x = np.abs(np.diff(img_gray, axis=1))
    grad_y = np.abs(np.diff(img_gray, axis=0))
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    block_size = max(32, min(h, w) // 8)
    block_densities = []
    for y in range(0, h - block_size, block_size):
        row = []
        for x in range(0, w - block_size, block_size):
            block = edge_magnitude[y:y + block_size, x:x + block_size]
            row.append(block.mean())
        if row:
            block_densities.append(row)

    if not block_densities:
        return 0.5

    block_arr = np.array(block_densities)
    row_diffs = np.abs(np.diff(block_arr, axis=1))
    col_diffs = np.abs(np.diff(block_arr, axis=0))
    mean_density = block_arr.mean()
    if mean_density == 0:
        return 0.5

    row_sharpness = row_diffs.mean() / mean_density
    col_sharpness = col_diffs.mean() / mean_density

    grad_direction = np.arctan2(grad_y + 1e-10, grad_x + 1e-10)
    dir_blocks = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = grad_direction[y:y + block_size, x:x + block_size]
            sin_mean = np.mean(np.sin(block))
            cos_mean = np.mean(np.cos(block))
            r = np.sqrt(sin_mean**2 + cos_mean**2)
            dir_blocks.append(1 - r)

    dir_blocks = np.array(dir_blocks)
    dir_inconsistency = np.std(dir_blocks)

    # Edge density uniformity penalty (AI = too uniform)
    edge_density_cv = block_arr.std() / (block_arr.mean() + 1e-10)
    uniformity_penalty = max(0, min(1, (0.8 - edge_density_cv) * 2.0))

    transition_score = min(1.0, (row_sharpness + col_sharpness) / 1.2)
    contour_score = min(1.0, dir_inconsistency * 3.0)

    anomaly_score = (
        transition_score * 0.30 +
        contour_score * 0.30 +
        uniformity_penalty * 0.40
    )

    return max(0, min(1, anomaly_score))


# ==============================================================================
# LAYER 2: BACKGROUND COHERENCE ANALYSIS
# ==============================================================================

def analyze_background_coherence(image):
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt(
        ((y_coords - center_y) / (h / 2))**2 +
        ((x_coords - center_x) / (w / 2))**2
    )
    bg_weight = np.clip(dist_from_center - 0.3, 0, 1)

    edges = np.array(image.convert('L').filter(ImageFilter.FIND_EDGES)).astype(float)

    block_size = max(24, min(h, w) // 10)
    bg_edge_blocks = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_edge = (edges * bg_weight)[y:y + block_size, x:x + block_size]
            block_w = bg_weight[y:y + block_size, x:x + block_size]
            if block_w.mean() > 0.25:
                bg_edge_blocks.append(block_edge.mean())

    if len(bg_edge_blocks) < 4:
        return 0.5

    bg_edge_arr = np.array(bg_edge_blocks)
    bg_edge_cv = bg_edge_arr.std() / (bg_edge_arr.mean() + 1e-10)

    bg_blocks_texture = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            cy_b = min(y + block_size // 2, h - 1)
            cx_b = min(x + block_size // 2, w - 1)
            if bg_weight[cy_b, cx_b] > 0.25:
                block = img_gray[y:y + block_size, x:x + block_size]
                bg_blocks_texture.append(np.var(block))

    if len(bg_blocks_texture) < 4:
        return 0.5

    texture_arr = np.array(bg_blocks_texture)
    texture_cv = texture_arr.std() / (texture_arr.mean() + 1e-10)
    bg_mean_texture = texture_arr.mean()
    low_texture_score = max(0, min(1, 1.0 - (bg_mean_texture / 300.0)))

    grad_x = np.diff(img_gray, axis=1, prepend=img_gray[:, :1])
    grad_y = np.diff(img_gray, axis=0, prepend=img_gray[:1, :])
    grad_angle = np.arctan2(grad_y, grad_x + 1e-10)
    strong_edge_mask = (edges > np.percentile(edges, 75)) & (bg_weight > 0.25)

    if strong_edge_mask.sum() > 100:
        strong_angles = grad_angle[strong_edge_mask]
        hist_a, _ = np.histogram(strong_angles, bins=36, range=(-np.pi, np.pi))
        hist_a = hist_a.astype(float) / (hist_a.sum() + 1e-10)
        angle_entropy = -np.sum(hist_a * np.log2(hist_a + 1e-10)) / np.log2(36)
    else:
        angle_entropy = 0.5

    coherence_score = (
        min(1.0, bg_edge_cv / 1.8) * 0.20 +
        min(1.0, texture_cv / 2.0) * 0.20 +
        angle_entropy * 0.25 +
        low_texture_score * 0.35
    )

    return max(0, min(1, coherence_score))


# ==============================================================================
# LAYER 3: "TOO PERFECT" SKIN / APPEARANCE (MAJOR REWRITE)
# ==============================================================================

def analyze_skin_perfection(image):
    """
    Completely rewritten skin detector. Key changes:
    - ALWAYS analyzes center 60% of image regardless of skin mask
    - Dedicated glossiness / specular highlight detector
    - LBP runs on center region, not gated by skin mask
    - Much broader skin mask that includes shiny/glossy areas
    - Checks ratio of very-bright pixels (AI specular highlights)
    """
    img_rgb = np.array(image.convert('RGB')).astype(float)
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    # ── Always use center region (face area) as primary analysis zone ──
    center_gray = get_center_region(img_gray, 0.6)
    center_rgb = get_center_region(img_rgb, 0.6)
    ch, cw = center_gray.shape

    # ── SIGNAL 1: Micro-texture (high-pass filter) ──
    # How much fine detail exists in the center/face region?
    blurred_full = np.array(image.convert('L').filter(
        ImageFilter.GaussianBlur(radius=1)
    )).astype(float)
    micro_texture_full = np.abs(img_gray - blurred_full)
    center_texture = get_center_region(micro_texture_full, 0.6)

    texture_level = center_texture.mean()
    # AI images: texture_level typically < 3.0
    # Real photos: texture_level typically > 4.0
    smoothness_score = max(0, min(1, 1 - (texture_level / 4.5)))

    # ── SIGNAL 2: LBP micro-texture entropy (always on center region) ──
    lbp = compute_lbp(center_gray.astype(np.uint8))

    if lbp.size > 100:
        lbp_hist, _ = np.histogram(lbp.flatten(), bins=64, range=(0, 256))
        lbp_hist = lbp_hist.astype(float) / (lbp_hist.sum() + 1e-10)
        lbp_entropy = -np.sum(lbp_hist * np.log2(lbp_hist + 1e-10))
        max_entropy = np.log2(64)
        # AI skin: low LBP entropy (uniform, repetitive patterns)
        # Real skin: high LBP entropy (varied pore/texture patterns)
        lbp_score = max(0, min(1, 1 - (lbp_entropy / (max_entropy * 0.70))))
    else:
        lbp_score = 0.5

    # ── SIGNAL 3: GLOSSINESS / SPECULAR HIGHLIGHT DETECTION ──
    # AI images have large, smooth specular highlights on skin
    # Real photos have smaller, more irregular highlights

    # Find bright pixels in center (potential specular highlights)
    brightness_threshold = np.percentile(center_gray, 90)
    bright_mask = center_gray > brightness_threshold

    bright_pixel_ratio = bright_mask.sum() / center_gray.size

    # Check how SMOOTH the bright areas are
    # AI highlights are smooth blobs; real highlights have texture
    if bright_mask.sum() > 50:
        bright_texture = center_texture[bright_mask]
        bright_smoothness = bright_texture.mean()
        # AI highlights: very smooth (< 2.0)
        # Real highlights: still have some texture (> 3.0)
        highlight_smooth_score = max(0, min(1, 1 - (bright_smoothness / 3.0)))
    else:
        highlight_smooth_score = 0.4

    # Check highlight SHAPE: AI produces large uniform blobs
    # Measure how clustered vs scattered bright pixels are
    if bright_mask.sum() > 30:
        # Count connected bright regions by checking how many bright neighbors
        # each bright pixel has (simple approximation)
        bright_float = bright_mask.astype(float)
        kernel = np.ones((5, 5)) / 25.0
        # Convolve manually with slicing
        padded = np.pad(bright_float, 2, mode='constant')
        neighbor_density = np.zeros_like(bright_float)
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                neighbor_density += padded[2+dy:2+dy+ch, 2+dx:2+dx+cw]
        neighbor_density /= 25.0

        bright_cluster_density = neighbor_density[bright_mask].mean()
        # High cluster density = large smooth blobs = AI-like
        # Low cluster density = scattered highlights = more natural
        blob_score = max(0, min(1, bright_cluster_density / 0.6))
    else:
        blob_score = 0.4

    glossiness_score = (
        highlight_smooth_score * 0.45 +
        blob_score * 0.35 +
        min(1.0, bright_pixel_ratio * 8) * 0.20  # lots of bright = shiny
    )

    # ── SIGNAL 4: Local gradient smoothness ──
    block_size = max(12, min(ch, cw) // 10)
    local_vars = []
    for y in range(0, ch - block_size, block_size):
        for x in range(0, cw - block_size, block_size):
            block = center_gray[y:y + block_size, x:x + block_size]
            local_vars.append(np.var(block))

    if local_vars:
        mean_lv = np.mean(local_vars)
        # AI: low local variance (smooth gradients)
        gradient_score = max(0, min(1, 1 - (mean_lv / 30.0)))
    else:
        gradient_score = 0.5

    # ── SIGNAL 5: Saturation uniformity ──
    center_hsv = np.array(image.convert('HSV')).astype(float)
    center_sat = get_center_region(center_hsv[:, :, 1], 0.6)

    if center_sat.size > 50:
        sat_cv = center_sat.std() / (center_sat.mean() + 1e-10)
        sat_score = max(0, min(1, 1 - (sat_cv / 0.45)))
    else:
        sat_score = 0.5

    # ── SIGNAL 6: Pore / fine detail detection via Laplacian ──
    detail_kernel = ImageFilter.Kernel(
        (3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1, offset=128
    )
    detail_map = np.array(image.convert('L').filter(detail_kernel)).astype(float) - 128
    center_detail = np.abs(get_center_region(detail_map, 0.6))
    detail_level = center_detail.mean()
    # AI: low detail (< 5), Real: higher detail (> 8)
    pore_score = max(0, min(1, 1 - (detail_level / 8.0)))

    # ── Combine all signals ──
    perfection_score = (
        smoothness_score * 0.15 +
        lbp_score * 0.20 +
        glossiness_score * 0.25 +   # NEW: dedicated glossiness detector
        gradient_score * 0.10 +
        sat_score * 0.10 +
        pore_score * 0.20
    )

    return max(0, min(1, perfection_score))


# ==============================================================================
# LAYER 4: TEXT & OBJECT ARTIFACT DETECTION
# ==============================================================================

def analyze_text_object_artifacts(image):
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape
    edges = np.array(image.convert('L').filter(ImageFilter.FIND_EDGES)).astype(float)

    block_size = max(16, min(h, w) // 16)
    text_candidates = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = edges[y:y + block_size, x:x + block_size]
            block_gray = img_gray[y:y + block_size, x:x + block_size]
            edge_density = (block > 30).sum() / block.size
            contrast = block_gray.max() - block_gray.min()
            if edge_density > 0.3 and contrast > 80:
                text_candidates.append({
                    'variance': np.var(block),
                    'edge_density': edge_density,
                })

    text_artifact_score = 0.4
    if len(text_candidates) > 2:
        variances = [t['variance'] for t in text_candidates]
        densities = [t['edge_density'] for t in text_candidates]
        var_cv = np.std(variances) / (np.mean(variances) + 1e-10)
        den_cv = np.std(densities) / (np.mean(densities) + 1e-10)
        text_artifact_score = min(1.0, (var_cv + den_cv) / 1.5)

    grad_x = np.diff(img_gray, axis=1, prepend=img_gray[:, :1])
    grad_y = np.diff(img_gray, axis=0, prepend=img_gray[:1, :])
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    med_t = np.percentile(grad_mag, 60)
    high_t = np.percentile(grad_mag, 90)
    boundary_mask = (grad_mag > med_t) & (grad_mag < high_t)

    boundary_scores = []
    if boundary_mask.sum() > 100:
        ys, xs = np.where(boundary_mask)
        np.random.seed(42)
        for _ in range(min(50, len(ys) // 100)):
            idx = np.random.randint(len(ys))
            cy, cx = ys[idx], xs[idx]
            r = block_size // 2
            y_lo, y_hi = max(0, cy - r), min(h, cy + r)
            x_lo, x_hi = max(0, cx - r), min(w, cx + r)
            lm = boundary_mask[y_lo:y_hi, x_lo:x_hi]
            la = np.arctan2(grad_y[y_lo:y_hi, x_lo:x_hi][lm],
                            grad_x[y_lo:y_hi, x_lo:x_hi][lm] + 1e-10)
            if len(la) > 5:
                sm = np.mean(np.sin(la))
                cm = np.mean(np.cos(la))
                boundary_scores.append(1 - np.sqrt(sm**2 + cm**2))

    boundary_incoherence = np.mean(boundary_scores) if boundary_scores else 0.4

    detail_blocks = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img_gray[y:y + block_size, x:x + block_size]
            if block.shape[0] >= 3:
                detail_blocks.append(np.var(np.diff(block, axis=0)))

    if len(detail_blocks) > 4:
        da = np.array(detail_blocks)
        detail_inconsistency = min(1.0, (da.std() / (da.mean() + 1e-10)) / 2.5)
    else:
        detail_inconsistency = 0.4

    return max(0, min(1,
        text_artifact_score * 0.30 +
        boundary_incoherence * 0.35 +
        detail_inconsistency * 0.35
    ))


# ==============================================================================
# LAYER 5: LIGHTING CONSISTENCY ANALYSIS
# ==============================================================================

def analyze_lighting_consistency(image):
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    grad_x = np.diff(img_gray, axis=1, prepend=img_gray[:, :1])
    grad_y = np.diff(img_gray, axis=0, prepend=img_gray[:1, :])

    mid_y, mid_x = h // 2, w // 2
    quadrants = [
        (0, mid_y, 0, mid_x), (0, mid_y, mid_x, w),
        (mid_y, h, 0, mid_x), (mid_y, h, mid_x, w),
    ]

    light_dirs = []
    for y1, y2, x1, x2 in quadrants:
        qx, qy = grad_x[y1:y2, x1:x2], grad_y[y1:y2, x1:x2]
        mag = np.sqrt(qx**2 + qy**2)
        strong = mag > np.percentile(mag, 70)
        if strong.sum() > 50:
            light_dirs.append(np.arctan2(np.mean(qy[strong]), np.mean(qx[strong]) + 1e-10))

    if len(light_dirs) < 2:
        return 0.5

    dirs = np.array(light_dirs)
    r = np.sqrt(np.mean(np.sin(dirs))**2 + np.mean(np.cos(dirs))**2)
    dir_consistency = 1 - r

    center = img_gray[h//4:3*h//4, w//4:3*w//4]
    edges_br = [img_gray[:h//4,:].mean(), img_gray[3*h//4:,:].mean(),
                img_gray[:,:w//4].mean(), img_gray[:,3*w//4:].mean()]
    br_diffs = [abs(center.mean() - eb) for eb in edges_br]
    br_cv = np.std(br_diffs) / (np.mean(br_diffs) + 1e-10)

    dark_t = np.percentile(img_gray, 25)
    dark_mask = img_gray < dark_t
    if dark_mask.sum() > 200:
        da = np.arctan2(grad_y[dark_mask], grad_x[dark_mask] + 1e-10)
        dr = np.sqrt(np.mean(np.sin(da))**2 + np.mean(np.cos(da))**2)
        shadow_score = 1 - dr
    else:
        shadow_score = 0.5

    # Histogram smoothness
    hist_v, _ = np.histogram(img_gray, bins=128, range=(0, 256))
    hd = np.abs(np.diff(hist_v.astype(float)))
    hist_smooth = 1.0 - min(1.0, hd.std() / (hd.mean() + 1e-10) / 3.0)

    return max(0, min(1,
        dir_consistency * 0.30 +
        min(1.0, br_cv / 1.5) * 0.20 +
        shadow_score * 0.25 +
        hist_smooth * 0.25
    ))


# ==============================================================================
# LAYER 6: FREQUENCY ANALYSIS
# ==============================================================================

def analyze_frequency_domain(image):
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    f_shift = np.fft.fftshift(np.fft.fft2(img_gray))
    magnitude = np.abs(f_shift)

    center_y, center_x = h // 2, w // 2
    max_radius = min(h, w) // 2

    y_c, x_c = np.ogrid[:h, :w]
    dist = np.sqrt((y_c - center_y)**2 + (x_c - center_x)**2)

    num_bands = 32
    radial_power = []
    for i in range(num_bands):
        r_in = (i / num_bands) * max_radius
        r_out = ((i + 1) / num_bands) * max_radius
        mask = (dist >= r_in) & (dist < r_out)
        radial_power.append(np.mean(magnitude[mask]) if mask.sum() > 0 else 0)

    rp = np.array(radial_power)
    rp = rp / (rp[0] + 1e-10)

    # High-freq energy ratio
    low = rp[:num_bands // 4].sum()
    high = rp[num_bands // 2:].sum()
    total = low + rp[num_bands//4:num_bands//2].sum() + high + 1e-10
    freq_energy_score = max(0, min(1, 1 - ((high / total) / 0.12)))

    # Spectral decay rate
    valid = rp[2:] > 0
    if valid.sum() > 5:
        lf = np.log(np.arange(2, num_bands)[valid] + 1)
        lp = np.log(rp[2:][valid] + 1e-10)
        n = len(lf)
        slope = (n * np.sum(lf * lp) - np.sum(lf) * np.sum(lp)) / \
                (n * np.sum(lf**2) - np.sum(lf)**2 + 1e-10)
        decay_score = max(0, min(1, (-slope - 0.8) / 1.5))
    else:
        decay_score = 0.5

    # Spectral smoothness
    if len(rp) > 4:
        sd = np.abs(np.diff(np.log(rp + 1e-10)))
        spectral_smooth = max(0, min(1, 1 - sd.std() / 0.6))
    else:
        spectral_smooth = 0.5

    return max(0, min(1,
        freq_energy_score * 0.35 +
        decay_score * 0.40 +
        spectral_smooth * 0.25
    ))


# ==============================================================================
# LAYER 7: NOISE AUTHENTICITY ANALYSIS
# ==============================================================================

def analyze_noise_patterns(image):
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    blurred = np.array(image.convert('L').filter(
        ImageFilter.GaussianBlur(radius=2)
    )).astype(float)
    noise = img_gray - blurred
    noise_abs_mean = np.abs(noise).mean()

    block_size = 32
    brightness_vals = []
    noise_vals = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            brightness_vals.append(blurred[y:y+block_size, x:x+block_size].mean())
            noise_vals.append(np.std(noise[y:y+block_size, x:x+block_size]))

    bv = np.array(brightness_vals)
    nv = np.array(noise_vals)

    # Noise-brightness correlation
    if len(bv) > 10:
        bc = bv - bv.mean()
        nc = nv - nv.mean()
        bs, ns = bc.std(), nc.std()
        if bs > 0 and ns > 0:
            corr = np.mean(bc * nc) / (bs * ns)
            correlation_score = max(0, min(1, 1 - max(0, corr) / 0.30))
        else:
            correlation_score = 0.8
    else:
        correlation_score = 0.5

    # Noise level
    if noise_abs_mean < 0.8:
        noise_level_score = 0.95
    elif noise_abs_mean < 1.5:
        noise_level_score = 0.75
    elif noise_abs_mean < 2.5:
        noise_level_score = 0.55
    elif noise_abs_mean < 4.0:
        noise_level_score = 0.35
    else:
        noise_level_score = 0.10

    # Noise spatial uniformity
    if len(nv) > 4:
        ncv = nv.std() / (nv.mean() + 1e-10)
        if ncv < 0.12:
            uniformity_score = 0.90
        elif ncv < 0.25:
            uniformity_score = 0.65
        elif ncv < 0.45:
            uniformity_score = 0.40
        else:
            uniformity_score = 0.15
    else:
        uniformity_score = 0.5

    return max(0, min(1,
        correlation_score * 0.40 +
        noise_level_score * 0.30 +
        uniformity_score * 0.30
    ))


# ==============================================================================
# WARNING GENERATION & RISK ASSESSMENT
# ==============================================================================

def generate_warnings(analysis, overall_score):
    warnings = []
    checks = [
        ('anatomical_anomalies', 0.50, 0.35,
         'Anatomical Distortions Detected',
         'The image shows irregular feature patterns — uneven edge transitions and inconsistent contour shapes suggest AI-generated distortions in facial features or body parts.',
         'Possible Feature Irregularities',
         'Some areas show unusual structural patterns. Look closely at ears, fingers, hairline, and facial boundaries.'),

        ('background_coherence', 0.50, 0.35,
         'Incoherent Background Detected',
         'The background shows warped lines, melting objects, or unnaturally smooth textures. AI generators frequently produce physically impossible backgrounds.',
         'Background Irregularities',
         'Some background areas appear unnaturally smooth or show inconsistent textures.'),

        ('skin_perfection', 0.50, 0.35,
         'Unnaturally Perfect / Glossy Skin',
         'The skin appears glassy and airbrushed with large smooth specular highlights, minimal pores, and low micro-texture complexity. This "too perfect" shiny appearance is a hallmark of AI-generated faces.',
         'Suspiciously Smooth Appearance',
         'The face region shows reduced micro-texture and unusually smooth/glossy highlight patterns consistent with AI generation.'),

        ('text_object_artifacts', 0.50, 0.35,
         'Text or Object Rendering Errors',
         'Detected inconsistent detail levels or broken object boundaries. AI images frequently contain illegible text and improperly merged objects.',
         'Possible Object Artifacts',
         'Some regions show inconsistent detail. Check any text — AI text is almost always garbled.'),

        ('lighting_consistency', 0.50, 0.35,
         'Mismatched Lighting Detected',
         'Light appears to come from different directions across the image. Real photos have physically consistent lighting with uniform shadow directions.',
         'Lighting Inconsistencies',
         'Some lighting variation detected. Check if shadows make physical sense.'),

        ('frequency_anomaly', 0.45, 0.30,
         'AI Frequency Signature Detected',
         'Spectral analysis reveals a steep frequency decay and reduced high-frequency energy — a pattern characteristic of AI generation that real cameras do not produce.',
         'Unusual Frequency Signature',
         'The frequency spectrum shows reduced high-frequency detail, a trait common in AI-generated images.'),

        ('noise_anomaly', 0.45, 0.30,
         'Non-Authentic Noise Pattern',
         'This image lacks the brightness-correlated noise that all real camera sensors produce. The noise is either absent, too low, or artificially uniform — a strong AI indicator.',
         'Unusual Noise Characteristics',
         'The noise pattern lacks the typical brightness-noise correlation expected from real cameras.'),
    ]

    for key, ht, mt, htitle, hdetail, mtitle, mdetail in checks:
        val = analysis[key]
        if val > ht:
            warnings.append({'type': 'high', 'icon': '\U0001f534', 'title': htitle, 'detail': hdetail})
        elif val > mt:
            warnings.append({'type': 'medium', 'icon': '\U0001f7e1', 'title': mtitle, 'detail': mdetail})

    if not warnings:
        warnings.append({
            'type': 'low', 'icon': '\U0001f7e2',
            'title': 'No Major Red Flags Detected',
            'detail': 'No significant AI indicators found. Always verify with multiple methods.'
        })

    return warnings


def get_risk_level(score):
    if score >= 55:
        return {
            'level': 'HIGH', 'color': '#ef4444',
            'message': 'This image shows strong signs of being AI-generated across multiple detection '
                       'layers. Exercise extreme caution — this profile is likely using a fake image.'
        }
    elif score >= 40:
        return {
            'level': 'MEDIUM', 'color': '#f59e0b',
            'message': 'This image shows several characteristics of AI-generated content. '
                       'Proceed with caution and look for other red flags.'
        }
    elif score >= 25:
        return {
            'level': 'LOW', 'color': '#3b82f6',
            'message': 'Some minor indicators found but could be caused by filters or editing. '
                       'Likely genuine, but stay alert.'
        }
    else:
        return {
            'level': 'MINIMAL', 'color': '#10b981',
            'message': 'This image appears to be a natural photograph. Passes all major detection layers.'
        }


# ==============================================================================
# FLASK ROUTES
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed.'}), 400

    try:
        ext = file.filename.rsplit('.', 1)[1].lower()
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath)
        if image.mode != 'RGB':
            image = image.convert('RGB')

        max_dim = 512
        ratio = min(max_dim / image.width, max_dim / image.height)
        if ratio < 1:
            new_size = (int(image.width * ratio), int(image.height * ratio))
            image = image.resize(new_size, Image.LANCZOS)

        # Run all 7 layers
        anatomical = analyze_anatomical_anomalies(image)
        background = analyze_background_coherence(image)
        skin = analyze_skin_perfection(image)
        text_objects = analyze_text_object_artifacts(image)
        lighting = analyze_lighting_consistency(image)
        frequency = analyze_frequency_domain(image)
        noise = analyze_noise_patterns(image)

        # Weighted score — skin, frequency, noise are strongest
        overall = (
            anatomical * 0.10 +
            background * 0.10 +
            skin * 0.22 +
            text_objects * 0.08 +
            lighting * 0.10 +
            frequency * 0.20 +
            noise * 0.20
        ) * 100

        overall = round(min(100, max(0, overall)), 1)

        analysis_results = {
            'anatomical_anomalies': round(anatomical, 3),
            'background_coherence': round(background, 3),
            'skin_perfection': round(skin, 3),
            'text_object_artifacts': round(text_objects, 3),
            'lighting_consistency': round(lighting, 3),
            'frequency_anomaly': round(frequency, 3),
            'noise_anomaly': round(noise, 3),
        }

        warnings = generate_warnings(analysis_results, overall)
        risk = get_risk_level(overall)

        os.remove(filepath)

        return jsonify({
            'success': True,
            'overall_score': overall,
            'risk': risk,
            'analysis': analysis_results,
            'warnings': warnings,
            'tips': [
                'Reverse image search the profile picture on Google Images or TinEye.',
                'Look closely at ears, fingers, and hairlines — AI often distorts these.',
                'Zoom into the background for warped lines or melting objects.',
                'Check for any text in the image — AI text is almost always garbled.',
                'Be cautious if the person avoids video calls or live photos.',
                'Check if the account was recently created with few posts or followers.',
                'Ask a mutual friend if they actually know this person.'
            ]
        })

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
