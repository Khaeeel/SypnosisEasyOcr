
import os
import json
import cv2
import re
import numpy as np
from difflib import SequenceMatcher, get_close_matches
import easyocr

# ----------------------------
# CONFIGURATION
# ----------------------------
IMAGE_FOLDER = "sample image"
OUTPUT_JSON = "easyocr_output.json"
OCR_CONF_THRESHOLD = 0.2
IMAGE_GAP_THRESHOLD = 150 

# --- INITIALIZE EASYOCR ENGINE ---
print("--- Initializing EasyOCR (CPU Mode) ---")
reader = easyocr.Reader(['en'], gpu=False)

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

    if not text: 
        return None
    
    clean = text.lower().strip()
    # Check for exact alias match first (Highest priority)
    if clean in SENDER_ALIASES:
        return SENDER_ALIASES[clean]
    
    # Direct match
    for alias, official in SENDER_ALIASES.items():
        if alias == clean:
            return official
    
    # Substring match (most common OCR error recovery)
    for alias, official in SENDER_ALIASES.items():
        if alias in clean or clean in alias:
            return official
    
    # Fuzzy matching
    matches = get_close_matches(clean, SENDER_ALIASES.keys(), n=1, cutoff=0.7)
    return SENDER_ALIASES[matches[0]] if matches else None

def is_timestamp(text):
    if not text:
        return False

    processed = text.upper().replace('O', '0').replace('I', '1').replace('.', ':')

    return bool(re.match(
        r'^(\d{1,2}:\d{2}|\d{3,4})\s?(AM|PM|M)$',
        processed
    ))
def clean_message_text(text):
    if not text:
        return ""

    stripped = text.strip()

    # HARD BLOCK date separators (Viber / chat UI labels)
    if stripped.lower() in {
        "yesterday",
        "today",
        "tomorrow",
        "monday", "tuesday", "wednesday",
        "thursday", "friday", "saturday", "sunday"
    }:
        return ""

    # Full date formats like "Tuesday, January 20, 2026"
    date_pattern = r'^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)(,?\s+\w+\s+\d{1,2},?\s+\d{4})?$'
    if re.match(date_pattern, stripped, flags=re.IGNORECASE):
        return ""

    # Reaction / icon artifacts
    if stripped in {"1", "2", "3"}:
        return ""

    text = remove_inline_timestamps(stripped)
    return text


def remove_inline_timestamps(text):
    if not text:
        return text

    # Skip if text is purely numeric (preserve like "123")
    if text.strip().isdigit():
        return text

    # Pattern 1: Proper timestamps with colons/periods (HH:MM AM/PM or HH.MM AM/PM)
    # Matches: 2:49 PM, 12:30 AM, 2.49pm, etc.
    pattern1 = r'\b([01]?\d|2[0-3])[:.]([0-5]\d)\s*(AM|PM|am|pm)?\b'
    text = re.sub(pattern1, '', text, flags=re.IGNORECASE)
    
    # Pattern 2: Malformed timestamps (OCR error) - digits directly followed by AM/PM with optional space
    # Matches: 249PM, 1230 AM, 3PM, 2 PM, etc.
    # This catches timestamps whether they have space or not
    pattern2 = r'\s*\d{1,4}\s*(AM|PM|am|pm)\s*'
    text = re.sub(pattern2, ' ', text, flags=re.IGNORECASE)
    
    # Pattern 3: Edge case - single digit followed by AM/PM at end of text
    # Matches: "something 2PM" -> "something"
    pattern3 = r'\s+\d{1,2}(AM|PM|am|pm)$'
    text = re.sub(pattern3, '', text, flags=re.IGNORECASE)

    # Cleanup extra spaces
    return re.sub(r'\s{2,}', ' ', text).strip()


def detect_dynamic_header(lines):
    """Find pinned message header and return Y position"""
    pinned_keywords = ["pinned on", "pinned message"]
    for line in lines:
        text = line[1][0].lower()  # Converted format: [bbox, [text, confidence]]
        if any(k in text for k in pinned_keywords):
            print(f"   -> Found Pinned Header. Ignoring content above.")
            return True  # Mark that we found a header
    return False

def is_noise(text):
    """Filter out system noise messages"""
    noise_triggers = ["activate windows", "go to settings", "type a message", "rakuten viber", "gif", "pinned on"]
    return any(trigger in text.lower() for trigger in noise_triggers)

def group_text_blocks(lines, threshold=15):
    """Merge lines that are very close to each other vertically."""
    if not lines: return []
    grouped = []
    current_block = lines[0]

    for next_line in lines[1:]:
        # If the gap between current bottom and next top is small
        curr_y_bottom = max(p[1] for p in current_block[0])
        next_y_top = min(p[1] for p in next_line[0])
        
        if (next_y_top - curr_y_bottom) < threshold:
            # Merge text and update bounding box to encompass both
            current_block[1][0] += " " + next_line[1][0]
            # Average the confidence
            current_block[1][1] = (current_block[1][1] + next_line[1][1]) / 2
        else:
            grouped.append(current_block)
            current_block = next_line
    grouped.append(current_block)
    return grouped

def get_ocr_results(img):
    """
    Get OCR results from EasyOCR and convert to PaddleOCR-like format for compatibility.
    EasyOCR output: [(text, confidence), ...]
    PaddleOCR-like format: [[bbox], [text, confidence]]
    """
    result = reader.readtext(img)
    
    if not result:
        # Fallback: Denoise if first attempt fails
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
        result = reader.readtext(enhanced)
    
    # Convert EasyOCR output to PaddleOCR-like format
    # PaddleOCR: [[bbox_points], [text, confidence]]
    # where bbox_points = [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    converted = []
    for item in result:
        bbox = item[0]  # List of 4 points
        text = item[1]
        conf = item[2]
        converted.append([bbox, [text, conf]])
    
    return converted

# ----------------------------
# MAIN LOGIC
# ----------------------------

def is_short_reply(text):
    return len(text.strip()) <= 4

def extract_chat():
    current_sender = None 
    current_sender_raw = None
    image_files = sorted([f for f in os.listdir(IMAGE_FOLDER) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    final_data = []
    
    for img_file in image_files:
        print(f"Processing {img_file}...")
        img = cv2.imread(os.path.join(IMAGE_FOLDER, img_file))
        if img is None: 
            continue

        lines_raw = get_ocr_results(img)
        if not lines_raw: 
            continue
        
        messages_data = []

        # Filter by confidence threshold and sort by Y coordinate
        lines = sorted(
            [line for line in lines_raw if line[1][1] > OCR_CONF_THRESHOLD], 
            key=lambda x: x[0][0][1]  # Sort by Y coordinate of first bbox point
        )
        
        header_cut_y = detect_dynamic_header(lines)
        last_y_bottom = 0 
        
        i = 0
        while i < len(lines):
            line_data = lines[i]
            bbox = line_data[0]
            text = line_data[1][0].strip()
            
            # --- CAPTURE CONFIDENCE HERE ---
            confidence = line_data[1][1]
            
            # Get coordinates from bbox
            y_coords = [p[1] for p in bbox]
            x_coords = [p[0] for p in bbox]
            y_top = min(y_coords)
            y_bottom = max(y_coords)
            x_left = min(x_coords)

            if is_noise(text):
                i += 1
                continue

            # Image Gap Detection
            reference_y = last_y_bottom if last_y_bottom > 0 else 0
            if current_sender and reference_y > 0 and (y_top - reference_y) > IMAGE_GAP_THRESHOLD:
                print(f"   -> Detected large gap ({y_top - reference_y}px). Inserting [PHOTO MESSAGE].")
                messages_data.append({
                    "timestamp": None,
                    "sender_raw": current_sender_raw,
                    "sender": current_sender, 
                    "message": "[PHOTO MESSAGE]",
                    "message_type": "text",
                    "confidence_score": 1.0,
                    "screenshot_path": img_file
                })

            if "photo message" in text.lower():
                messages_data.append({
                    "timestamp": None,
                    "sender_raw": current_sender_raw if current_sender_raw else "Unknown",
                    "sender": current_sender if current_sender else "Unknown",
                    "message": "[PHOTO MESSAGE]",
                    "message_type": "text",
                    "confidence_score": float(confidence),
                    "screenshot_path": img_file
                })
                last_y_bottom = y_bottom
                i += 1
                continue

            cleaned = clean_message_text(text)

            # Preserve very short but meaningful replies (Ty, Yes, Ok, Oo nga)
            if cleaned == "" and len(text.strip()) <= 5:
                cleaned = text.strip()

            if cleaned == "" and text != "":
                i += 1
                continue
            
            last_y_bottom = y_bottom

            name1 = resolve_sender(cleaned)
            name1_raw = cleaned  # Store raw sender name
            name2 = None
            name2_raw = None
            if i + 1 < len(lines):
                next_text = clean_message_text(lines[i + 1][1][0].strip())
                name2 = resolve_sender(next_text)
                name2_raw = next_text

            # --- SCENARIO A: REPLY BLOCK ---
            if name1 and name2 and name1 != name2:
                current_sender = name1
                current_sender_raw = name1_raw
                target_sender = name2
                
                # Get X positions for indent threshold
                sender_x = x_left
                next_bbox = lines[i + 1][0]
                target_x = min([p[0] for p in next_bbox])
                indent_threshold = (sender_x + target_x) / 2
                
                quote_parts = []
                reply_parts = []
                scores = [confidence, lines[i + 1][1][1]]
                
                k = i + 2
                while k < len(lines):
                    next_line = lines[k]
                    next_bbox = next_line[0]
                    next_text = clean_message_text(next_line[1][0].strip())
                    next_conf = next_line[1][1]
                    next_x = min([p[0] for p in next_bbox])
                    next_y_coords = [p[1] for p in next_bbox]
                    next_y_top = min(next_y_coords)
                    
                    if is_noise(next_line[1][0]): 
                        k += 1
                        continue
                    if is_timestamp(next_text): 
                        if next_x > indent_threshold: 
                            k += 1
                            continue 
                        else: 
                            k += 1
                            break 
                    if resolve_sender(next_text): 
                        break

                    scores.append(next_conf)
                    if next_x > indent_threshold: 
                        quote_parts.append(next_text)
                    else: 
                        reply_parts.append(next_text)
                    last_y_bottom = max(next_y_coords)
                    k += 1
                
                full_quote = " ".join(quote_parts) if quote_parts else "[MEDIA/LINK PREVIEW]"
                full_reply = " ".join(reply_parts)
                if not full_reply and "[PHOTO MESSAGE]" in str(final_data[-1] if final_data else ""): 
                    full_reply = "[PHOTO MESSAGE]"

                # Avg Confidence for block
                avg_conf = sum(scores) / len(scores) if scores else 0.0

                messages_data.append({
                    "timestamp": None,
                    "sender_raw": current_sender_raw,
                    "sender": current_sender, 
                    "message": full_reply,
                    "message_type": "reply",
                    "confidence_score": round(avg_conf, 4),
                    "screenshot_path": img_file,
                    "reply_to": {
                        "original_sender": target_sender, 
                        "original_message": full_quote
                    }
                })
                i = k 
                continue

            # --- SCENARIO B: STANDARD MESSAGE ---
            elif name1:
                current_sender = name1
                current_sender_raw = name1_raw
                i += 1
                continue

            # --- SCENARIO C: TIMESTAMP ---
            elif is_timestamp(cleaned):
                if messages_data: 
                    messages_data[-1]["timestamp"] = cleaned
                i += 1
                continue

            # --- SCENARIO D: CONTINUATION ---
            else:
                if current_sender and cleaned:
                    last_msg = messages_data[-1] if messages_data else None
                    if last_msg and last_msg["sender"] == current_sender:
                        
                        if last_msg["message"] == "[PHOTO MESSAGE]":
                            messages_data.append({
                                "timestamp": None,
                                "sender_raw": current_sender_raw,
                                "sender": current_sender, 
                                "message": cleaned, 
                                "message_type": "text",
                                "confidence_score": float(confidence),
                                "screenshot_path": img_file
                            })
                        else:
                            last_msg["message"] += f" {cleaned}"
                            # Merge confidence scores
                            old_conf = last_msg.get("confidence_score", 0.0)
                            last_msg["confidence_score"] = round((old_conf + confidence) / 2, 4)
                    else:
                        messages_data.append({
                            "timestamp": None,
                            "sender_raw": current_sender_raw,
                            "sender": current_sender, 
                            "message": cleaned, 
                            "message_type": "text",
                            "confidence_score": float(confidence),
                            "screenshot_path": img_file
                        })
                i += 1

        # Add file object to final_data if messages exist
        if messages_data:
            cleaned_messages = [d for d in messages_data if d['message'].strip() != "" and d['message'] != "Pinned on"]
            if cleaned_messages:
                final_data.append({
                    "file": img_file,
                    "messages": cleaned_messages
                })

    # Save to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(final_data, f, indent=2, ensure_ascii=False)
    print(f"Extraction complete. Saved to {OUTPUT_JSON}")

if __name__ == "__main__":
    extract_chat()
