"""
CatfishGuard - AI-Powered Catfishing Detection Tool
Backend server that analyzes profile images for signs of AI generation.

Detection Layers:
  1. Anatomical Anomalies   - Irregular features, distorted proportions, warped edges
  2. Background Coherence   - Warped/incoherent backgrounds, melting objects
  3. Skin Perfection        - Unnaturally smooth, glassy, poreless skin
  4. Text & Object Artifacts- Garbled text, incoherent object boundaries
  5. Lighting Consistency   - Mismatched lighting between subject and background
  6. Frequency Domain       - Missing natural high-frequency camera detail
  7. Noise Pattern Analysis - Absent or artificial sensor noise
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


# ==============================================================================
# LAYER 1: ANATOMICAL ANOMALY DETECTION
# ==============================================================================

def analyze_anatomical_anomalies(image):
    """
    Detects anatomical distortions common in AI-generated images:
    - Irregular edge density across facial regions (warped ears, jawlines)
    - Abnormal local gradient patterns (twisted fingers, misshapen features)
    - Inconsistent contour smoothness between body regions
    """
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    # Compute local edge density map using gradients
    grad_x = np.abs(np.diff(img_gray, axis=1))
    grad_y = np.abs(np.diff(img_gray, axis=0))
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
    edge_magnitude = np.sqrt(grad_x**2 + grad_y**2)

    # Divide into grid blocks and compute edge density per block
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
        return 0.3

    block_arr = np.array(block_densities)

    # Look for abnormal density transitions (warped features cause sudden jumps)
    row_diffs = np.abs(np.diff(block_arr, axis=1))
    col_diffs = np.abs(np.diff(block_arr, axis=0))

    mean_density = block_arr.mean()
    if mean_density == 0:
        return 0.3

    row_sharpness = row_diffs.mean() / mean_density
    col_sharpness = col_diffs.mean() / mean_density

    # Check gradient direction consistency (distortions cause erratic changes)
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

    transition_score = min(1.0, (row_sharpness + col_sharpness) / 1.5)
    contour_score = min(1.0, dir_inconsistency * 3.0)
    anomaly_score = transition_score * 0.55 + contour_score * 0.45

    return max(0, min(1, anomaly_score))


# ==============================================================================
# LAYER 2: BACKGROUND COHERENCE ANALYSIS
# ==============================================================================

def analyze_background_coherence(image):
    """
    Detects background errors typical of AI generation:
    - Warped straight lines, melting object boundaries
    - Abrupt texture changes in background regions
    - Nonsensical spatial relationships
    """
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    # Center-weighted mask to separate background from foreground
    center_y, center_x = h // 2, w // 2
    y_coords, x_coords = np.ogrid[:h, :w]
    dist_from_center = np.sqrt(
        ((y_coords - center_y) / (h / 2))**2 +
        ((x_coords - center_x) / (w / 2))**2
    )
    bg_weight = np.clip(dist_from_center - 0.4, 0, 1)

    edges = np.array(image.convert('L').filter(ImageFilter.FIND_EDGES)).astype(float)
    bg_edges = edges * bg_weight

    block_size = max(24, min(h, w) // 10)
    bg_edge_blocks = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_edge = bg_edges[y:y + block_size, x:x + block_size]
            block_w = bg_weight[y:y + block_size, x:x + block_size]
            if block_w.mean() > 0.3:
                bg_edge_blocks.append(block_edge.mean())

    if len(bg_edge_blocks) < 4:
        return 0.3

    bg_edge_arr = np.array(bg_edge_blocks)
    bg_edge_cv = bg_edge_arr.std() / (bg_edge_arr.mean() + 1e-10)

    # Background texture incoherence
    bg_blocks_texture = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            if bg_weight[y + block_size // 2, x + block_size // 2] > 0.3:
                block = img_gray[y:y + block_size, x:x + block_size]
                bg_blocks_texture.append(np.var(block))

    if len(bg_blocks_texture) < 4:
        return 0.3

    texture_arr = np.array(bg_blocks_texture)
    texture_cv = texture_arr.std() / (texture_arr.mean() + 1e-10)

    # Line warping detection via gradient direction entropy
    grad_x = np.diff(img_gray, axis=1, prepend=img_gray[:, :1])
    grad_y = np.diff(img_gray, axis=0, prepend=img_gray[:1, :])
    grad_angle = np.arctan2(grad_y, grad_x + 1e-10)
    strong_edge_mask = (edges > np.percentile(edges, 75)) & (bg_weight > 0.3)

    if strong_edge_mask.sum() > 100:
        strong_angles = grad_angle[strong_edge_mask]
        hist, _ = np.histogram(strong_angles, bins=36, range=(-np.pi, np.pi))
        hist = hist.astype(float) / (hist.sum() + 1e-10)
        angle_entropy = -np.sum(hist * np.log2(hist + 1e-10)) / np.log2(36)
    else:
        angle_entropy = 0.5

    edge_incoherence = min(1.0, bg_edge_cv / 2.0)
    texture_incoherence = min(1.0, texture_cv / 2.5)

    coherence_score = (
        edge_incoherence * 0.35 +
        texture_incoherence * 0.30 +
        angle_entropy * 0.35
    )

    return max(0, min(1, coherence_score))


# ==============================================================================
# LAYER 3: "TOO PERFECT" SKIN / APPEARANCE
# ==============================================================================

def analyze_skin_perfection(image):
    """
    Detects unnaturally perfect skin and appearance:
    - Airbrushed/glassy skin lacking pores and blemishes
    - Excessively smooth gradients in skin-tone regions
    - Missing micro-texture that all real skin has
    """
    img_rgb = np.array(image.convert('RGB')).astype(float)
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]

    # Broad skin-tone detection across many skin colors
    skin_mask = (
        (r > 50) & (g > 30) & (b > 15) &
        (r < 250) & (g < 240) & (b < 230) &
        (r > g) & (r > b) &
        (np.abs(r - g) > 5) &
        ((r - b) > 10)
    )

    skin_pixels = skin_mask.sum()
    total_pixels = h * w
    use_skin = skin_pixels >= total_pixels * 0.05

    # Measure micro-texture via high-pass filter
    blurred = np.array(image.convert('L').filter(
        ImageFilter.GaussianBlur(radius=1)
    )).astype(float)
    micro_texture = np.abs(img_gray - blurred)

    if use_skin:
        skin_texture = micro_texture[skin_mask]
    else:
        cy, cx = h // 2, w // 2
        rh, rw = h // 4, w // 4
        skin_texture = micro_texture[cy - rh:cy + rh, cx - rw:cx + rw].flatten()

    if len(skin_texture) == 0:
        return 0.3

    texture_level = skin_texture.mean()
    smoothness_score = max(0, min(1, 1 - (texture_level / 8.0)))

    # Check gradient smoothness in skin regions
    block_size = max(16, min(h, w) // 12)
    local_vars = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block_mask = skin_mask[y:y + block_size, x:x + block_size]
            if block_mask.sum() > block_size * block_size * 0.5:
                block = img_gray[y:y + block_size, x:x + block_size]
                block_vals = block[block_mask]
                if len(block_vals) > 10:
                    local_vars.append(np.var(block_vals))

    if local_vars:
        mean_lv = np.mean(local_vars)
        gradient_score = max(0, min(1, 1 - (mean_lv / 50.0)))
    else:
        gradient_score = 0.3

    # Detect absence of skin detail (pores, blemishes) via Laplacian
    detail_kernel = ImageFilter.Kernel(
        (3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1], scale=1, offset=128
    )
    detail_map = np.array(image.convert('L').filter(detail_kernel)).astype(float) - 128

    if use_skin:
        skin_detail = np.abs(detail_map[skin_mask])
    else:
        cy, cx = h // 2, w // 2
        rh, rw = h // 4, w // 4
        skin_detail = np.abs(detail_map[cy - rh:cy + rh, cx - rw:cx + rw].flatten())

    detail_level = skin_detail.mean() if len(skin_detail) > 0 else 5
    pore_score = max(0, min(1, 1 - (detail_level / 12.0)))

    perfection_score = (
        smoothness_score * 0.40 +
        gradient_score * 0.30 +
        pore_score * 0.30
    )

    return max(0, min(1, perfection_score))


# ==============================================================================
# LAYER 4: TEXT & OBJECT ARTIFACT DETECTION
# ==============================================================================

def analyze_text_object_artifacts(image):
    """
    Detects text and object rendering errors:
    - Garbled, scrambled, or illegible text-like patterns
    - Incoherent object boundaries where things merge
    - Inconsistent detail levels between adjacent objects
    """
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    edges = np.array(image.convert('L').filter(ImageFilter.FIND_EDGES)).astype(float)

    block_size = max(16, min(h, w) // 16)
    text_candidates = []
    regular_blocks = []

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
            else:
                regular_blocks.append({'variance': np.var(block)})

    # Analyze text region coherence
    text_artifact_score = 0.3
    if len(text_candidates) > 2:
        variances = [t['variance'] for t in text_candidates]
        densities = [t['edge_density'] for t in text_candidates]
        var_consistency = np.std(variances) / (np.mean(variances) + 1e-10)
        density_consistency = np.std(densities) / (np.mean(densities) + 1e-10)
        text_artifact_score = min(1.0, (var_consistency + density_consistency) / 2.0)

    # Object boundary incoherence
    grad_x = np.diff(img_gray, axis=1, prepend=img_gray[:, :1])
    grad_y = np.diff(img_gray, axis=0, prepend=img_gray[:1, :])
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    med_threshold = np.percentile(grad_mag, 60)
    high_threshold = np.percentile(grad_mag, 90)
    boundary_mask = (grad_mag > med_threshold) & (grad_mag < high_threshold)

    boundary_consistency_scores = []
    if boundary_mask.sum() > 100:
        ys, xs = np.where(boundary_mask)
        np.random.seed(42)
        sample_count = min(50, len(ys) // 100)
        for _ in range(sample_count):
            idx = np.random.randint(len(ys))
            cy, cx = ys[idx], xs[idx]
            r = block_size // 2
            y_lo, y_hi = max(0, cy - r), min(h, cy + r)
            x_lo, x_hi = max(0, cx - r), min(w, cx + r)

            local_mask = boundary_mask[y_lo:y_hi, x_lo:x_hi]
            local_angles = np.arctan2(
                grad_y[y_lo:y_hi, x_lo:x_hi][local_mask],
                grad_x[y_lo:y_hi, x_lo:x_hi][local_mask] + 1e-10
            )

            if len(local_angles) > 5:
                sin_m = np.mean(np.sin(local_angles))
                cos_m = np.mean(np.cos(local_angles))
                r_val = np.sqrt(sin_m**2 + cos_m**2)
                boundary_consistency_scores.append(1 - r_val)

    boundary_incoherence = np.mean(boundary_consistency_scores) if boundary_consistency_scores else 0.3

    # Detail level inconsistency across blocks
    detail_blocks = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = img_gray[y:y + block_size, x:x + block_size]
            if block.shape[0] >= 3 and block.shape[1] >= 3:
                detail_blocks.append(np.var(np.diff(block, axis=0)))

    if len(detail_blocks) > 4:
        detail_arr = np.array(detail_blocks)
        detail_cv = detail_arr.std() / (detail_arr.mean() + 1e-10)
        detail_inconsistency = min(1.0, detail_cv / 3.0)
    else:
        detail_inconsistency = 0.3

    artifact_score = (
        text_artifact_score * 0.30 +
        boundary_incoherence * 0.35 +
        detail_inconsistency * 0.35
    )

    return max(0, min(1, artifact_score))


# ==============================================================================
# LAYER 5: LIGHTING CONSISTENCY ANALYSIS
# ==============================================================================

def analyze_lighting_consistency(image):
    """
    Detects lighting inconsistencies between subject and background:
    - Mismatched light direction across image quadrants
    - Inconsistent brightness gradients
    - Unrealistic shadow directions
    """
    img_gray = np.array(image.convert('L')).astype(float)
    h, w = img_gray.shape

    grad_x = np.diff(img_gray, axis=1, prepend=img_gray[:, :1])
    grad_y = np.diff(img_gray, axis=0, prepend=img_gray[:1, :])

    # Estimate light direction in each quadrant
    mid_y, mid_x = h // 2, w // 2
    quadrants = [
        (0, mid_y, 0, mid_x),
        (0, mid_y, mid_x, w),
        (mid_y, h, 0, mid_x),
        (mid_y, h, mid_x, w),
    ]

    light_directions = []
    for y1, y2, x1, x2 in quadrants:
        qx = grad_x[y1:y2, x1:x2]
        qy = grad_y[y1:y2, x1:x2]
        mag = np.sqrt(qx**2 + qy**2)
        strong = mag > np.percentile(mag, 70)

        if strong.sum() > 50:
            avg_dx = np.mean(qx[strong])
            avg_dy = np.mean(qy[strong])
            direction = np.arctan2(avg_dy, avg_dx + 1e-10)
            light_directions.append(direction)

    if len(light_directions) < 2:
        return 0.3

    # Check light direction consistency across quadrants
    directions = np.array(light_directions)
    sin_mean = np.mean(np.sin(directions))
    cos_mean = np.mean(np.cos(directions))
    r = np.sqrt(sin_mean**2 + cos_mean**2)
    direction_consistency = 1 - r

    # Compare center vs periphery brightness
    center_region = img_gray[h // 4:3 * h // 4, w // 4:3 * w // 4]
    edge_regions = [
        img_gray[:h // 4, :],
        img_gray[3 * h // 4:, :],
        img_gray[:, :w // 4],
        img_gray[:, 3 * w // 4:]
    ]

    center_brightness = center_region.mean()
    edge_brightnesses = [r.mean() for r in edge_regions]
    brightness_diffs = [abs(center_brightness - eb) for eb in edge_brightnesses]
    brightness_cv = np.std(brightness_diffs) / (np.mean(brightness_diffs) + 1e-10)
    brightness_score = min(1.0, brightness_cv / 2.0)

    # Shadow consistency
    dark_threshold = np.percentile(img_gray, 25)
    dark_mask = img_gray < dark_threshold
    if dark_mask.sum() > 200:
        dark_angles = np.arctan2(grad_y[dark_mask], grad_x[dark_mask] + 1e-10)
        dark_sin = np.mean(np.sin(dark_angles))
        dark_cos = np.mean(np.cos(dark_angles))
        dark_r = np.sqrt(dark_sin**2 + dark_cos**2)
        shadow_consistency = 1 - dark_r
    else:
        shadow_consistency = 0.3

    lighting_score = (
        direction_consistency * 0.40 +
        brightness_score * 0.30 +
        shadow_consistency * 0.30
    )

    return max(0, min(1, lighting_score))


# ==============================================================================
# LAYER 6: FREQUENCY DOMAIN ANALYSIS
# ==============================================================================

def analyze_frequency_domain(image):
    """
    AI-generated images lack the natural high-frequency noise
    present in real camera photos from sensor properties and optics.
    """
    img_gray = np.array(image.convert('L')).astype(float)

    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shift))

    h, w = magnitude.shape
    center_y, center_x = h // 2, w // 2

    radius_low = min(h, w) // 8
    radius_high = min(h, w) // 3

    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)

    low_energy = magnitude[dist <= radius_low].sum()
    high_energy = magnitude[dist >= radius_high].sum()

    if low_energy == 0:
        return 0.5

    freq_ratio = high_energy / low_energy
    freq_score = max(0, min(1, 1 - (freq_ratio / 2.0)))

    return freq_score


# ==============================================================================
# LAYER 7: NOISE PATTERN ANALYSIS
# ==============================================================================

def analyze_noise_patterns(image):
    """
    Real camera photos have sensor noise that varies with brightness.
    AI images either lack noise or have artificial uniform noise.
    """
    img_gray = np.array(image.convert('L')).astype(float)

    blurred = np.array(image.convert('L').filter(
        ImageFilter.GaussianBlur(radius=2)
    )).astype(float)
    noise = img_gray - blurred

    noise_std = noise.std()

    block_size = 64
    h, w = noise.shape
    noise_stds = []

    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise[y:y + block_size, x:x + block_size]
            noise_stds.append(block.std())

    if not noise_stds:
        return 0.5

    noise_stds = np.array(noise_stds)
    noise_cv = noise_stds.std() / (noise_stds.mean() + 1e-10)

    if noise_std < 1.5:
        noise_level_score = 0.85
    elif noise_std < 3.0:
        noise_level_score = 0.6
    else:
        noise_level_score = 0.2

    if noise_cv < 0.2:
        uniformity_score = 0.7
    elif noise_cv < 0.4:
        uniformity_score = 0.5
    else:
        uniformity_score = 0.2

    return max(0, min(1, noise_level_score * 0.5 + uniformity_score * 0.5))


# ==============================================================================
# WARNING GENERATION & RISK ASSESSMENT
# ==============================================================================

def generate_warnings(analysis, overall_score):
    warnings = []

    checks = [
        ('anatomical_anomalies', 0.65, 0.50,
         'Anatomical Distortions Detected',
         'The image shows irregular feature patterns — uneven edge transitions and inconsistent contour shapes that suggest warped or distorted body features (ears, jawline, fingers, eyes). Real photos don\'t exhibit these artifacts.',
         'Possible Feature Irregularities',
         'Some areas show unusual structural patterns. Look closely at ears, fingers, hairline, and facial boundaries for signs of warping or distortion.'),

        ('background_coherence', 0.65, 0.50,
         'Incoherent Background Detected',
         'The background shows signs of warped lines, melting objects, or inconsistent textures. AI generators frequently produce backgrounds where straight lines curve and objects blend together unnaturally.',
         'Background Irregularities',
         'Some background areas show unusual texture or edge patterns. Check for warped door frames, blurry nonsensical objects, or areas that seem to melt together.'),

        ('skin_perfection', 0.70, 0.50,
         'Unnaturally Perfect Skin',
         'The skin appears glassy and airbrushed with almost no visible pores, blemishes, or natural texture. All real human skin retains some micro-texture that this image lacks.',
         'Suspiciously Smooth Appearance',
         'Skin areas appear smoother than typical photographs. While beauty filters can cause this, this level of smoothness could indicate AI generation.'),

        ('text_object_artifacts', 0.65, 0.50,
         'Text or Object Rendering Errors',
         'Detected patterns consistent with garbled text, broken object boundaries, or inconsistent detail levels. AI images often contain illegible text and objects that merge together.',
         'Possible Object Artifacts',
         'Some regions show inconsistent detail levels. Look for any text in the image — AI-generated text is almost always scrambled or misspelled.'),

        ('lighting_consistency', 0.65, 0.50,
         'Mismatched Lighting Detected',
         'Light appears to come from different directions in different parts of the image. In real photos, all shadows point the same way. This mismatch is a strong indicator of AI generation.',
         'Lighting Inconsistencies',
         'The lighting direction shows some variation across the image. Check if shadows and highlights make physical sense with a single light source.'),

        ('frequency_anomaly', 0.65, 0.50,
         'Missing Natural Frequency Patterns',
         'The image lacks the high-frequency details normally found in camera photos. Real cameras produce characteristic frequency signatures that AI generators cannot replicate.',
         'Unusual Frequency Signature',
         'The frequency distribution differs somewhat from typical camera photos.'),

        ('noise_anomaly', 0.65, 0.50,
         'Abnormal Noise Pattern',
         'This image either lacks natural camera sensor noise or has artificially uniform noise. Real photos always contain subtle grain that varies with brightness.',
         'Unusual Noise Characteristics',
         'The noise pattern doesn\'t fully match typical camera sensor behavior.'),
    ]

    for key, high_thresh, med_thresh, high_title, high_detail, med_title, med_detail in checks:
        val = analysis[key]
        if val > high_thresh:
            warnings.append({'type': 'high', 'icon': '\U0001f534', 'title': high_title, 'detail': high_detail})
        elif val > med_thresh:
            warnings.append({'type': 'medium', 'icon': '\U0001f7e1', 'title': med_title, 'detail': med_detail})

    if not warnings:
        warnings.append({
            'type': 'low', 'icon': '\U0001f7e2',
            'title': 'No Major Red Flags Detected',
            'detail': 'No significant indicators of AI generation were found across our 7 detection layers. '
                      'However, AI generators are improving — always use multiple verification methods.'
        })

    return warnings


def get_risk_level(score):
    if score >= 70:
        return {
            'level': 'HIGH', 'color': '#ef4444',
            'message': 'This image shows strong signs of being AI-generated across multiple detection '
                       'layers. Exercise extreme caution — this profile is likely using a fake image.'
        }
    elif score >= 50:
        return {
            'level': 'MEDIUM', 'color': '#f59e0b',
            'message': 'This image shows several characteristics of AI-generated content. '
                       'Proceed with caution and look for other red flags like new accounts or limited posts.'
        }
    elif score >= 30:
        return {
            'level': 'LOW', 'color': '#3b82f6',
            'message': 'A few minor indicators were found, but they could be caused by filters or editing. '
                       'The profile is likely genuine, but stay alert.'
        }
    else:
        return {
            'level': 'MINIMAL', 'color': '#10b981',
            'message': 'This image appears to be a natural photograph. It passes all major detection layers '
                       'with no significant AI generation indicators.'
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
        return jsonify({'error': 'File type not allowed. Use PNG, JPG, JPEG, GIF, WEBP, or BMP.'}), 400

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

        # Run all 7 detection layers
        anatomical = analyze_anatomical_anomalies(image)
        background = analyze_background_coherence(image)
        skin = analyze_skin_perfection(image)
        text_objects = analyze_text_object_artifacts(image)
        lighting = analyze_lighting_consistency(image)
        frequency = analyze_frequency_domain(image)
        noise = analyze_noise_patterns(image)

        # Weighted overall score
        overall = (
            anatomical * 0.18 +
            background * 0.15 +
            skin * 0.18 +
            text_objects * 0.12 +
            lighting * 0.15 +
            frequency * 0.12 +
            noise * 0.10
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
                'Zoom into the background for warped lines, melting objects, or nonsensical scenes.',
                'Check for any text in the image — AI-generated text is almost always garbled.',
                'Be cautious if the person avoids video calls or sending live, unposed photos.',
                'Check if the account was recently created with very few posts or followers.',
                'Ask a mutual friend if they actually know this person in real life.'
            ]
        })

    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500


@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
