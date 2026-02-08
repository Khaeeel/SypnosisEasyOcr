import os
import json
import cv2
import re
import numpy as np
from difflib import SequenceMatcher, get_close_matches
from paddleocr import PaddleOCR

# ----------------------------
# CONFIGURATION
# ----------------------------
IMAGE_FOLDER = "images"
OUTPUT_JSON = "paddle_output.json" # Saved as paddle_output for comparison
OCR_CONF_THRESHOLD = 0.3
IMAGE_GAP_THRESHOLD = 150 

# --- FORCE CPU TO PREVENT CRASHES ---
# We use the old syntax because it works well with the logic you already have.
print("--- Initializing PaddleOCR (CPU Mode) ---")
ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False)

SENDER_ALIASES = {
    "dom": "Dominic Almazan", "doms": "Dominic Almazan", "domi": "Dominic Almazan", "dominic": "Dominic Almazan",
    "leonardo yoro": "Leonardo Yoro", "uno yoro": "Leonardo Yoro", "uno": "Leonardo Yoro", "leo": "Leonardo Yoro", "leonardo": "Leonardo Yoro",
    "naithan balonda": "Naithan Balondo", "naithan balondo": "Naithan Balondo", "naitnan baionda": "Naithan Balondo", "naithan": "Naithan Balonda",
    "bharon candelaria": "Bharon Candelaria", "bharon": "Bharon Candelaria", "lebron": "Lebron",
    "alberto catapang": "Alberto Catapang", "abet": "Alberto Catapang", "alberto": "Alberto Catapang",
    "JPA": "JPA", "jpa": "JPA",
    "NTS": "NTS", "nts": "NTS",
    "michael magsino": "Michael Magsino",
    "jonathan": "Jonathan",
    "sandy": "Sandy", "joan hechanova": "Joan Hechanova",
    "comfac-ems": "Comfac-EMS", "jo-ann p.m.": "Jo-Ann P.M.",
}

# ----------------------------
# HELPER FUNCTIONS  
# ----------------------------
def resolve_sender(text):
    if not text: return None
    clean = text.lower().strip().replace('ai', 'al').replace('0', 'o')
    for alias, official in SENDER_ALIASES.items():
        if alias in clean: return official
    matches = get_close_matches(clean, SENDER_ALIASES.keys(), n=1, cutoff=0.7)
    return SENDER_ALIASES[matches[0]] if matches else None

def is_timestamp(text):
    if not text: return False
    processed = text.upper().replace('O', '0').replace('I', '1').replace('.', ':')
    return bool(re.search(r'\d{1,2}:\d{2}\s?(AM|PM|M)', processed))

def clean_message_text(text):
    if not text: return ""
    text = re.sub(r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4}', '', text, flags=re.IGNORECASE)
    days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "today", "yesterday"]
    if text.lower().strip() in days: return ""
    text = text.replace("Photo message", "")
    return text.strip()

def detect_dynamic_header(lines):
    pinned_keywords = ["pinned on", "pinned message"]
    for line in lines:
        text = line[1][0].lower()
        if any(k in text for k in pinned_keywords):
            print(f"   -> Found Pinned Header. Ignoring content above Y={line[0][2][1]}")
            return line[0][2][1] + 5 
    return 0

def is_noise(text):
    noise_triggers = ["activate windows", "go to settings", "type a message", "rakuten viber", "gif", "pinned on"]
    return any(trigger in text.lower() for trigger in noise_triggers)

def get_ocr_results(img):
    # Wrapper to handle potential image issues
    result = ocr.ocr(img, cls=True)
    if result and result[0]: return result
    
    # Fallback: Denoise if first attempt fails
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    return ocr.ocr(enhanced, cls=True)

# ----------------------------
# MAIN LOGIC
# ----------------------------
def extract_chat():
    current_sender = None 
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    final_data = []
    
    for img_file in image_files:
        print(f"Processing {img_file}...")
        img = cv2.imread(os.path.join(IMAGE_FOLDER, img_file))
        if img is None: continue

        result = get_ocr_results(img)
        if not result or not result[0]: continue

        lines = sorted([line for line in result[0] if line[1][1] > OCR_CONF_THRESHOLD], key=lambda x: x[0][0][1])
        
        header_cut_y = detect_dynamic_header(lines)
        last_y_bottom = 0 
        
        i = 0
        while i < len(lines):
            line_data = lines[i]
            text = line_data[1][0].strip()
            
            # --- CAPTURE CONFIDENCE HERE ---
            confidence = line_data[1][1] 
            
            y_top = line_data[0][0][1]
            y_bottom = line_data[0][2][1]
            x_left = line_data[0][0][0]

            if y_bottom <= header_cut_y:
                i += 1; continue

            if is_noise(text):
                i += 1; continue

            # Image Gap Detection
            reference_y = last_y_bottom if last_y_bottom > 0 else header_cut_y
            if current_sender and reference_y > 0 and (y_top - reference_y) > IMAGE_GAP_THRESHOLD:
                print(f"   -> Detected large gap ({y_top - reference_y}px). Inserting [PHOTO MESSAGE].")
                final_data.append({
                    "sender": current_sender, "message": "[PHOTO MESSAGE]",
                    "timestamp": None, "is_reply": False, "reply_to": None,
                    "confidence_score": 1.0
                })

            if "photo message" in text.lower():
                final_data.append({
                    "sender": current_sender if current_sender else "Unknown",
                    "message": "[PHOTO MESSAGE]",
                    "timestamp": None, "is_reply": False, "reply_to": None,
                    "confidence_score": float(confidence)
                })
                last_y_bottom = y_bottom; i += 1; continue

            cleaned = clean_message_text(text)
            if cleaned == "" and text != "": 
                i += 1; continue
            
            last_y_bottom = y_bottom

            name1 = resolve_sender(cleaned)
            name2 = None
            if i + 1 < len(lines):
                name2 = resolve_sender(clean_message_text(lines[i+1][1][0].strip()))

            # --- SCENARIO A: REPLY BLOCK ---
            if name1 and name2 and name1 != name2:
                current_sender = name1
                target_sender = name2
                sender_x = x_left
                target_x = lines[i+1][0][0][0]
                indent_threshold = (sender_x + target_x) / 2
                
                quote_parts = []
                reply_parts = []
                scores = [confidence, lines[i+1][1][1]]
                
                k = i + 2
                while k < len(lines):
                    next_line = lines[k]
                    next_text = clean_message_text(next_line[1][0].strip())
                    next_conf = next_line[1][1]
                    next_x = next_line[0][0][0]
                    next_y_top = next_line[0][0][1]
                    
                    if next_y_top <= header_cut_y or is_noise(next_line[1][0]): k+=1; continue
                    if is_timestamp(next_text): 
                        if next_x > indent_threshold: k += 1; continue 
                        else: k += 1; break 
                    if resolve_sender(next_text): break

                    scores.append(next_conf)
                    if next_x > indent_threshold: quote_parts.append(next_text)
                    else: reply_parts.append(next_text)
                    last_y_bottom = next_line[0][2][1]; k += 1
                
                full_quote = " ".join(quote_parts) if quote_parts else "[MEDIA/LINK PREVIEW]"
                full_reply = " ".join(reply_parts)
                if not full_reply and "[PHOTO MESSAGE]" in str(final_data[-1]): full_reply = "[PHOTO MESSAGE]"

                # Avg Confidence for block
                avg_conf = sum(scores) / len(scores) if scores else 0.0

                final_data.append({
                    "sender": current_sender, "message": full_reply,
                    "timestamp": None, "is_reply": True,
                    "reply_to": {"original_sender": target_sender, "original_message": full_quote},
                    "confidence_score": round(avg_conf, 4)
                })
                i = k 
                continue

            # --- SCENARIO B: STANDARD MESSAGE ---
            elif name1:
                current_sender = name1; i += 1; continue

            # --- SCENARIO C: TIMESTAMP ---
            elif is_timestamp(cleaned):
                if final_data: final_data[-1]["timestamp"] = cleaned
                i += 1; continue

            # --- SCENARIO D: CONTINUATION ---
            else:
                if current_sender and cleaned:
                    last_msg = final_data[-1] if final_data else None
                    if last_msg and last_msg["sender"] == current_sender and not is_timestamp(last_msg.get("timestamp", "")):
                        
                        if last_msg["message"] == "[PHOTO MESSAGE]":
                             final_data.append({
                                "sender": current_sender, "message": cleaned, 
                                "timestamp": None, "is_reply": False, "reply_to": None,
                                "confidence_score": float(confidence)
                            })
                        else:
                            last_msg["message"] += f" {cleaned}"
                            # Merge confidence scores
                            old_conf = last_msg.get("confidence_score", 0.0)
                            last_msg["confidence_score"] = round((old_conf + confidence) / 2, 4)
                    else:
                        final_data.append({
                            "sender": current_sender, "message": cleaned, 
                            "timestamp": None, "is_reply": False, "reply_to": None,
                            "confidence_score": float(confidence)
                        })
                i += 1

    cleaned_data = [d for d in final_data if d['message'].strip() != "" and d['message'] != "Pinned on"]

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(cleaned_data, f, indent=2, ensure_ascii=False)
    print(f"Extraction complete. Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_chat()