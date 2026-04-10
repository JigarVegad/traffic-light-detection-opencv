import cv2
import numpy as np
import glob
import os

# ----------------------------
# Configuration
# ----------------------------
CONFIG = {
    'resize_width': 600,
    'min_area': 150,
    'max_area': 4000,
    'aspect_ratio_min': 0.7,
    'aspect_ratio_max': 1.3,
    'extent_min': 0.6,
    'circularity_min': 0.6,
    'solidity_min': 0.75,
    'brightness_min': 120,
    'saturation_min': 80,
    'hsv_ranges': {
        'RED': [
            ((0, 80, 100), (12, 255, 255)),
            ((160, 80, 100), (180, 255, 255))
        ],
        'YELLOW': [
            ((12, 80, 100), (40, 255, 255))
        ],
        'GREEN': [
            ((35, 70, 100), (95, 255, 255))
        ]
    }
}

# ----------------------------
# Resize helper
# ----------------------------
def resize_image(img, width):
    h, w = img.shape[:2]
    ratio = width / float(w)
    return cv2.resize(img, (width, int(h * ratio)))

# ----------------------------
# Detect lights
# ----------------------------
def detect_light(mask, hsv, color_name):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        area = cv2.contourArea(cnt)

        if area < CONFIG['min_area'] or area > CONFIG['max_area']:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        aspect_ratio = float(w) / h
        if not (CONFIG['aspect_ratio_min'] < aspect_ratio < CONFIG['aspect_ratio_max']):
            continue

        extent = area / float(w * h)
        if extent < CONFIG['extent_min']:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter == 0:
            continue

        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        if circularity < CONFIG['circularity_min']:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / float(hull_area) if hull_area > 0 else 0
        if solidity < CONFIG['solidity_min']:
            continue

        roi = hsv[y:y+h, x:x+w]
        brightness = np.mean(roi[:, :, 2])
        saturation = np.mean(roi[:, :, 1])

        if brightness < CONFIG['brightness_min'] or saturation < CONFIG['saturation_min']:
            continue

        confidence = (brightness / 255.0) * circularity * solidity

        detections.append((x, y, w, h, color_name, confidence))

    return detections

# ----------------------------
# Process image
# ----------------------------
def process_image(image_path):
    print(f"Processing: {image_path}")

    image = cv2.imread(image_path)
    if image is None:
        print("Error loading image")
        return

    image = resize_image(image, CONFIG['resize_width'])
    output = image.copy()

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Masks
    detections = []

    for color_name, ranges in CONFIG['hsv_ranges'].items():
        mask = None
        for lower, upper in ranges:
            m = cv2.inRange(hsv, lower, upper)
            mask = m if mask is None else cv2.bitwise_or(mask, m)

        detections += detect_light(mask, hsv, color_name)

    final_signal = "NO SIGNAL DETECTED"

    if detections:
        # pick best detection
        best = max(detections, key=lambda x: x[5])

        x, y, w, h, color, conf = best

        color_map = {
            "RED": (0, 0, 255),
            "YELLOW": (0, 255, 255),
            "GREEN": (0, 255, 0)
        }

        cv2.rectangle(output, (x, y), (x+w, y+h), color_map[color], 3)
        cv2.putText(output, f"{color} ({conf:.2f})", (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_map[color], 2)

        final_signal = f"{color} SIGNAL"

    # Display result
    cv2.putText(output, final_signal, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0, 255, 0), 2)

    # Save result
    os.makedirs("results", exist_ok=True)
    filename = os.path.basename(image_path)
    cv2.imwrite(f"results/{filename}", output)

    cv2.imshow("Traffic Light Detection", output)
    cv2.waitKey(0)

# ----------------------------
# MAIN
# ----------------------------
image_files = glob.glob("images/*")
image_files = [f for f in image_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

if not image_files:
    print("No images found!")
else:
    for img_path in image_files:
        process_image(img_path)

cv2.destroyAllWindows()