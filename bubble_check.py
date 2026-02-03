import cv2
import easyocr
import numpy as np
import os

# ------------------------
# CONFIG
# ------------------------
IMAGE_FOLDER = r"C:\Users\Dominic\OneDrive\Desktop\Sypnosis\sample image"
OUTPUT_FOLDER = r"C:\Users\Dominic\OneDrive\Desktop\Sypnosis\experiment output"
OCR_CONF_THRESHOLD = 0.

reader = easyocr.Reader(['en'], gpu=False)

# ------------------------
# MAIN BUBBLE DETECTION
# ------------------------
def detect_bubbles(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dark_mask = cv2.inRange(hsv, np.array([90, 0, 30]), np.array([140, 80, 160]))
    light_mask = cv2.inRange(hsv, np.array([0, 0, 180]), np.array([180, 50, 255]))

    mask = cv2.bitwise_or(dark_mask, light_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 80 and h > 40:
            bubbles.append((x, y, w, h))

    bubbles.sort(key=lambda b: b[1])
    return bubbles

# ------------------------
# SUB-BUBBLE (REPLY PREVIEW) DETECTION
# ------------------------
def detect_sub_bubbles(bubble_img):
    hsv = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2HSV)

    # Reply previews are lighter gray
    sub_mask = cv2.inRange(
        hsv,
        np.array([0, 0, 160]),
        np.array([180, 40, 240])
    )

    kernel = np.ones((3, 3), np.uint8)
    sub_mask = cv2.dilate(sub_mask, kernel, iterations=2)

    contours, _ = cv2.findContours(sub_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sub_bubbles = []
    h, w = bubble_img.shape[:2]

    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)

        # Must be smaller than main bubble and wide enough
        if bw > w * 0.4 and bh > 25 and bh < h * 0.6:
            sub_bubbles.append((x, y, bw, bh))

    sub_bubbles.sort(key=lambda b: b[1])
    return sub_bubbles

# ------------------------
# PROCESS IMAGES
# ------------------------
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg'))]

for img_file in image_files:
    img_path = os.path.join(IMAGE_FOLDER, img_file)
    img = cv2.imread(img_path)
    if img is None:
        continue

    bubbles = detect_bubbles(img)

    for x, y, w, h in bubbles:
        bubble_img = img[y:y+h, x:x+w]

        # Draw main bubble (GREEN)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # -------- SUB-BUBBLES --------
        sub_bubbles = detect_sub_bubbles(bubble_img)

        for sx, sy, sw, sh in sub_bubbles:
            # Draw sub-bubble (BLUE)
            cv2.rectangle(
                img,
                (x + sx, y + sy),
                (x + sx + sw, y + sy + sh),
                (255, 0, 0),
                2
            )

            # OCR sub-bubble
            sub_img = bubble_img[sy:sy+sh, sx:sx+sw]
            sub_ocr = reader.readtext(sub_img)

            for bbox, text, conf in sub_ocr:
                if conf < OCR_CONF_THRESHOLD:
                    continue
                pt = (int(bbox[0][0] + x + sx), int(bbox[0][1] + y + sy))
                cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # -------- MAIN TEXT OCR --------
        ocr_results = reader.readtext(bubble_img)
        for bbox, text, conf in ocr_results:
            if conf < OCR_CONF_THRESHOLD:
                continue
            pt = (int(bbox[0][0] + x), int(bbox[0][1] + y))
            cv2.putText(img, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    output_path = os.path.join(OUTPUT_FOLDER, img_file)
    cv2.imwrite(output_path, img)
    print(f"Processed {img_file} → saved to {output_path}")

print("All images processed!")
