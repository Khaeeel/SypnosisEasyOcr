import cv2
import easyocr
import re
import json
from datetime import datetime
import numpy as np
import os
from difflib import get_close_matches
from difflib import SequenceMatcher

reader = easyocr.Reader(['en'], gpu=False)

IMAGE_FOLDER = r"C:\Users\Dominic\OneDrive\Desktop\Sypnosis\sample image"
OCR_CONF_THRESHOLD = 0.0
V_GAP_THRESHOLD = 15

REJECT_WORDS = {
    'guys', 'hello', 'agree', 'yes', 'no', 'okay', 'ok', 'sure', 'thanks',
    'please', 'pls', 'lol', 'haha', 'hehe', 'wow', 'copy', 'good', 'bad',
    'and', 'the', 'from', 'like', 'that', 'this', 'what', 'when', 'where',
    'nice', 'great', 'done', 'wait', 'here', 'there', 'on', 'in', 'at'
    'Activate Windows Go to Settings to activate Windows olo',
    'Activate Windows',
    'Go to Settings to activate Windows', 'olo'
}

SENDER_ALIASES = {
    "dom": "Dominic Almazan",
    "doms": "Dominic Almazan",
    "domi": "Dominic Almazan",
    "dominic": "Dominic Almazan",
    "leonardo yoro": "Leonardo Yoro",
    "uno yoro": "Leonardo Yoro",
    "uno": "Leonardo Yoro",
    "leo": "Leonardo Yoro",
    "leonardo": "Leonardo Yoro",
    "naithan balonda": "Naithan Balondo",
    "naithan balondo": "Naithan Balondo",
    "naitnan baionda": "Naithan Balondo",
    "naithan": "Naithan Balonda",
    "bharon candelaria": "Bharon Candelaria",
    "bharon": "Bharon Candelaria",
    "lebron": "Lebron",
    "alberto catapang": "Alberto Catapang",
    "abet": "Alberto Catapang",
    "alberto": "Alberto Catapang",
    "JPA": "JPA",
    "jpa": "JPA",
    "NTS": "NTS",
    "nts": "NTS",
    "Sandy" : "Sandy",
    "Jonathan": "Jonathan",
    "Michael Magsino" : "Michael Magsino"
}

def is_grey_color(rgb_tuple, tolerance=30):
    if not rgb_tuple or len(rgb_tuple) < 3:
        return False
    
    r, g, b = rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]
    
    if not (110 <= r <= 200 and 110 <= g <= 200 and 110 <= b <= 200):
        return False
    
    return abs(r - g) <= tolerance and abs(g - b) <= tolerance and abs(r - b) <= tolerance

def is_violet_sender_color(rgb_tuple, tolerance=20):
    if not rgb_tuple or len(rgb_tuple) < 3:
        return False
    
    r, g, b = rgb_tuple[0], rgb_tuple[1], rgb_tuple[2]
    target_r, target_g, target_b = 142, 161, 255
    
    return (abs(r - target_r) <= tolerance and 
            abs(g - target_g) <= tolerance and 
            abs(b - target_b) <= tolerance)


def detect_bubbles(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    dark_mask = cv2.inRange(
        hsv,
        np.array([90, 0, 30]),
        np.array([140, 80, 160])
    )
    mask = dark_mask

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

def is_image_bubble(bubble_img, ocr_results):
    if not ocr_results:
        return False 

 
    avg_conf = np.mean([conf for _, _, conf in ocr_results])
    if avg_conf < 0.15:  
        return True

    gray = cv2.cvtColor(bubble_img, cv2.COLOR_BGR2GRAY)
    if np.std(gray) > 60:  
        return True

    return False


def get_text_region_color(img, bbox):
    if img is None or not bbox:
        return None
    
    try:
        x_coords = [point[0] for point in bbox]
        y_coords = [point[1] for point in bbox]
        x_min, x_max = int(min(x_coords)), int(max(x_coords))
        y_min, y_max = int(min(y_coords)), int(max(y_coords))
        
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(img.shape[1], x_max + padding)
        y_max = min(img.shape[0], y_max + padding)
        
        region = img[y_min:y_max, x_min:x_max]
        
        if region.size == 0:
            return None
        
        avg_color = cv2.mean(region)
        return (int(avg_color[2]), int(avg_color[1]), int(avg_color[0]))
    except:
        return None

def is_reply_preview_text(text, img=None, bbox=None):
    if not text:
        return False

    text_stripped = text.strip()

    timestamp_pattern = r'^\d{1,2}\s*[:.]?\s*\d{2}\s*(AM|PM|am|pm)?\s*$'
    if re.match(timestamp_pattern, text_stripped):
        return False

    system_exact_phrases = [
        'activate windows',
        'type message',
        'calculator',
        'message_',
        'message:',
        'go to settings',
        'pin', 'edit', 'delete', 'forward', 'share'
    ]
    text_lower = text_stripped.lower()
    for phrase in system_exact_phrases:
        if text_lower == phrase or text_lower.startswith(phrase + ' ') or text_lower.endswith(' ' + phrase):
            return False

    if len(text_stripped) > 80:
        return False

    if ':' in text_stripped:
        parts = text_stripped.split(':', 1)
        if len(parts) == 2:
            potential_sender = parts[0].strip()
            potential_preview = parts[1].strip()
            if 2 <= len(potential_sender) <= 25 and len(potential_preview) >= 2:
                return True

    # allow short acknowledgements as reply previews
    short_reply_words = {"ty", "thanks", "yes", "yeah", "oo", "nga", "ok", "okay", "sure"}

    if text_stripped.lower() in short_reply_words:
        return True

    if 2 <= len(text_stripped) <= 80:
        has_alpha = any(c.isalpha() for c in text_stripped)
        if has_alpha:
            return True

    return False


def is_reply_to_previous(current_text, recent_messages, current_sender=None, threshold=0.4):
        # VERY SHORT replies → attach to most recent different sender
    if len(current_text.strip()) <= 5:
        for msg in reversed(recent_messages[-5:]):
            if msg.get("sender") != current_sender and msg.get("message"):
                return {
                    "sender": msg.get("sender"),
                    "message_preview": msg.get("message", "")[:100]
                }
    if not current_text or not recent_messages:
        return None
    
    current_lower = current_text.lower().strip()
    
  
    if 'replying' in current_lower:
        for msg in reversed(recent_messages[-20:]):
            if msg.get("sender") != current_sender and msg.get("message"):
                return {
                    "sender": msg.get("sender"),
                    "message_preview": msg.get("message", "")[:100]
                }
    
   
    if '@' in current_text:
        for msg in reversed(recent_messages[-20:]):
            msg_sender = msg.get("sender")
            if current_sender and msg_sender == current_sender:
                continue
            if msg.get("message_type") not in ["system"] and msg.get("message"):
                return {
                    "sender": msg.get("sender"),
                    "message_preview": msg.get("message", "")[:100]
                }
    
    
    for msg in reversed(recent_messages[-10:]):
        prev_text = msg.get("message", "")
        if not prev_text or msg.get("sender") == current_sender:
            continue
        ratio = SequenceMatcher(None, current_text, prev_text).ratio()
        if ratio >= threshold:
            return {
                "sender": msg.get("sender"),
                "message_preview": prev_text[:100]
            }
    return None

def extract_reply_preview_data(text):
    if not text:
        return {}

    reply_data = {}
    text_orig = text
    text = text.strip().replace('|', '').replace('!', '')
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    if not lines:
        return {}

    first_line_lower = lines[0].lower()
    if first_line_lower in SENDER_ALIASES:
        sender = SENDER_ALIASES[first_line_lower]
        message_preview = " ".join(lines[1:]) if len(lines) > 1 else ""
        if sender and message_preview and len(message_preview) > 2:
            reply_data["sender"] = sender
            reply_data["message_preview"] = message_preview
            return reply_data

    if ':' in text:
        match = re.match(r'^([^:]+):\s*(.*)$', text)
        if match:
            potential_sender = match.group(1).strip()
            msg_content = match.group(2).strip()
            if 2 <= len(potential_sender) <= 25 and msg_content and len(msg_content) > 2:
                reply_data["sender"] = potential_sender
                reply_data["message_preview"] = msg_content
                return reply_data

    if 3 <= len(text) <= 80:
        has_alpha = any(c.isalpha() for c in text)
        if has_alpha:
            reply_data["message_preview"] = text
            return reply_data

    return {}

def is_system_message(text, img=None, bbox=None):
    if not text:
        return False
    
    text_lower = text.lower().strip()
    
    if img is not None and bbox is not None:
        text_color = get_text_region_color(img, bbox)
        if text_color and is_grey_color(text_color):
            if re.search(r'(joined|left|added|removed|new message)', text_lower):
                return True
            return False

    
    system_patterns = [
        r'^new message',
        r'^you (added?|removed|invite)',
        r'^(joined|left|leave) (the )?group',
        r'^group (created|updated)',
        r'^call (started|ended|missed)',
        r'^voice message',
        r'^photo',
        r'^video',
        r'^document',
        r'^location',
        r'^contact',
        r'^payment',
        r'^reaction',
        r'^\*.*\*$',
        r'\b(joined|left|was removed)(\s+(the )?group)?\s*$',
        r'^[a-z\s]+ (joined|left|was removed)',
    ]
    
    for pattern in system_patterns:
        if re.search(pattern, text_lower):
            return True
    
    if len(text_lower) < 50:
        system_keywords = [
            'new messages', 'added', 'removed', 'left the group', 'joined',
            'call started', 'call ended', 'call missed', 'voice message',
            'photo', 'video', 'document', 'location', 'contact', 'payment',
        ]
        for keyword in system_keywords:
            if keyword in text_lower:
                return True
    
    return False

def is_viber_date_separator(text, left_x, img_width, img=None, bbox=None):
    if not text:
        return False
    
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    
  
    date_patterns = [
        r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}[,\s]*\d{4}',
        r'^(today|yesterday|tomorrow)$',
        r'^(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}[,\s]*\d{4}',
        r'^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$',
    ]
    
  
    matches_date_pattern = any(re.search(pattern, text_lower) for pattern in date_patterns)
    
    if not matches_date_pattern:
        return False
    
    center_threshold_left = 0.35
    center_threshold_right = 0.65
    relative_position = left_x / img_width if img_width > 0 else 0
    
    is_centered = center_threshold_left <= relative_position <= center_threshold_right
    
    if not is_centered:
        return False
    

    if img is not None and bbox is not None:
        text_color = get_text_region_color(img, bbox)
        if text_color:
            if is_grey_color(text_color):
                return True

            return False
    return True

def fix_ocr_errors(text):
    if not text:
        return text
    
    if text.strip().isdigit():
        return text

    text = re.sub(r'\bl1(?=\w)', 'li', text)
    text = re.sub(r'(?<=\w)1l', 'il', text)
    text = re.sub(r'IIl', 'Ill', text)
    text = re.sub(r'II(?=\s)', 'II', text)
    text = re.sub(r'(?<=[A-Za-z])1(?=[A-Za-z])', 'I', text)

    return text


def extract_urls(text):
    if not text:
        return text, []
    
    url_pattern = r'https?://[^\s]+|www\.[^\s]+'
    urls = re.findall(url_pattern, text, re.IGNORECASE)
    cleaned_text = re.sub(url_pattern, '', text, flags=re.IGNORECASE).strip()
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)
    
    return cleaned_text, urls

def detect_reply_block(bubble_objects, idx):
    """
    Detect reply block pattern: sender1 -> quoted_lines -> sender2 -> reply_lines
    Returns: (reply_data, last_processed_idx) or (None, idx)
    """
    if idx + 1 >= len(bubble_objects):
        return None, idx
    
    current_bubble = bubble_objects[idx]
    next_bubble = bubble_objects[idx + 1]
    
    # Get sender names from current and next bubbles
    current_sender = None
    next_sender = None
    current_conf = 0
    next_conf = 0
    
    # Extract first text from current bubble
    if current_bubble["ocr"]:
        first_text = current_bubble["ocr"][0][1].strip() if current_bubble["ocr"][0][1] else ""
        current_sender = resolve_sender_alias(first_text) if is_valid_sender(first_text, current_bubble["x"], current_bubble["w"]) else None
        current_conf = current_bubble["ocr"][0][2] if len(current_bubble["ocr"][0]) > 2 else 0
    
    # Extract first text from next bubble
    if next_bubble["ocr"]:
        first_text = next_bubble["ocr"][0][1].strip() if next_bubble["ocr"][0][1] else ""
        next_sender = resolve_sender_alias(first_text) if is_valid_sender(first_text, next_bubble["x"], next_bubble["w"]) else None
        next_conf = next_bubble["ocr"][0][2] if len(next_bubble["ocr"][0]) > 2 else 0
    
    # Check if we have sender1 -> quoted_text -> sender2 pattern
    if not (current_sender and next_sender and current_sender != next_sender):
        return None, idx
    
    # Determine indent threshold for separating quoted from reply text
    sender_x = current_bubble["x"]
    target_x = next_bubble["x"]
    indent_threshold = (sender_x + target_x) / 2
    
    # Collect quote and reply parts
    quote_parts = []
    reply_parts = []
    scores = [current_conf, next_conf]
    
    # Process lines from current bubble (these are quoted)
    if current_bubble["ocr"] and len(current_bubble["ocr"]) > 1:
        for ocr_item in current_bubble["ocr"][1:]:
            text = ocr_item[1].strip() if ocr_item[1] else ""
            if text and not is_system_message(text):
                quote_parts.append(text)
                if len(ocr_item) > 2:
                    scores.append(ocr_item[2])
    
    # Process lines from next bubble - separate by indent
    if next_bubble["ocr"] and len(next_bubble["ocr"]) > 1:
        for ocr_item in next_bubble["ocr"][1:]:
            text = ocr_item[1].strip() if ocr_item[1] else ""
            if text and not is_system_message(text):
                # Approximate indent check
                x_pos = next_bubble["x"]
                if x_pos > indent_threshold:
                    quote_parts.append(text)
                else:
                    reply_parts.append(text)
                if len(ocr_item) > 2:
                    scores.append(ocr_item[2])
    
    full_quote = " ".join(quote_parts).strip() if quote_parts else "[MEDIA/LINK PREVIEW]"
    full_reply = " ".join(reply_parts).strip()
    
    if not full_reply:
        return None, idx
    
    avg_conf = round(float(np.mean(scores)), 4) if scores else 0.0
    
    reply_data = {
        "original_sender": next_sender,
        "original_message": full_quote
    }
    
    return {
        "reply_data": reply_data,
        "reply_sender": current_sender,
        "reply_text": full_reply,
        "confidence": avg_conf
    }, idx + 1

def format_timestamp(ts_str):
    if not ts_str:
        return None
    
    try:
        match = re.search(r'(\d{1,2}):(\d{2})\s*(AM|PM)', ts_str, re.IGNORECASE)
        if match:
            hour = int(match.group(1))
            minute = int(match.group(2))
            ampm = match.group(3).upper()
            
            if ampm == 'PM' and hour != 12:
                hour += 12
            elif ampm == 'AM' and hour == 12:
                hour = 0
            
            now_date = datetime.now().strftime("%Y-%m-%d")
            return f"{now_date} {hour:02d}:{minute:02d}"
    except:
        pass
    
    return ts_str

def normalize_sender(sender):
    if not sender:
        return sender
    
    sender = fix_ocr_errors(sender)
    sender = sender.title()
    
    return sender

def resolve_sender_alias(sender_raw):
    """
    Resolve sender name using PaddleOCR-style sophisticated matching:
    1. Direct match in aliases
    2. Substring match (handle OCR errors like 'ai' vs 'al', '0' vs 'o')
    3. Fuzzy matching with 0.7 cutoff
    """
    if not sender_raw:
        return sender_raw

    # Step 1: Clean and normalize text with common OCR errors
    clean = sender_raw.lower().strip()
    clean = clean.replace('ai', 'al').replace('0', 'o').replace('1', 'i').replace('l', 'i')
    
    # Step 2: Direct match in SENDER_ALIASES
    for alias, official in SENDER_ALIASES.items():
        if alias == clean:
            return official
    
    # Step 3: Substring/partial match (most common OCR error recovery)
    for alias, official in SENDER_ALIASES.items():
        if alias in clean or clean in alias:
            return official
    
    # Step 4: Fuzzy matching with high confidence threshold
    matches = get_close_matches(clean, SENDER_ALIASES.keys(), n=1, cutoff=0.7)
    if matches:
        return SENDER_ALIASES[matches[0]]

    return sender_raw

def should_merge(prev_msg, current_sender, current_text):
    if not prev_msg:
        return False

    if prev_msg["sender"] != current_sender:
        return False

    # timestamps / day markers should not merge
    if current_text.lower() in ["yesterday", "today"]:
        return False

    if (
        current_text[0].islower()
        or len(current_text.split()) <= 3
        or prev_msg["message"].endswith((",", ";"))
    ):
        return True

    return False


def post_process_message(msg, previous_sender=None):
    if not msg:
        return msg

    msg["message"] = fix_ocr_errors(msg["message"])
    msg["sender_raw"] = normalize_sender(msg.get("sender_raw", msg.get("sender", "")))
    resolved_sender = resolve_sender_alias(msg.get("sender_raw"))
    
    if msg.get("sender_raw") and resolved_sender:
        msg["sender"] = resolved_sender

    elif previous_sender:
        msg["sender"] = previous_sender
    else:
        msg["sender"] = "Unknown"

    previous_sender = msg["sender"]

    previous_sender = msg["sender"]
    msg["message"], urls = extract_urls(msg["message"])
    if urls:
        msg["urls"] = urls

    if msg.get("timestamp"):
        msg["timestamp"] = format_timestamp(msg["timestamp"])

    if msg.get("reply_to"):
        reply_to = msg["reply_to"]
        if "message_preview" in reply_to:
            reply_to["message_preview"] = fix_ocr_errors(reply_to["message_preview"])
            reply_to["message_preview"], _ = extract_urls(reply_to["message_preview"])
        if "sender" in reply_to:
            reply_to["sender"] = normalize_sender(reply_to["sender"])

    return msg, previous_sender


def infer_sender_from_text(message_text, previous_sender=None):
    if not message_text:
        return previous_sender or "Unknown"

    text_lower = message_text.lower()
    alias_counts = {}

    for key, full_name in SENDER_ALIASES.items():
        key_lower = key.lower()
        if key_lower in text_lower:
            alias_counts[full_name] = alias_counts.get(full_name, 0) + text_lower.count(key_lower)

    if alias_counts:
        return max(alias_counts, key=alias_counts.get)

    if previous_sender:
        return previous_sender

    return "Unknown"


def is_valid_sender(text, left_position, image_width, y_position=None, img=None, bbox=None, is_first_in_screenshot=False, state=None):
    """
    Enhanced sender validation using PaddleOCR logic:
    - Filters out system/action messages
    - Uses color detection (violet = sender)
    - Pattern matching for proper names
    - Position-based heuristics
    """
    if state and state.get("reply_preview_zone") and y_position is not None:
        zone = state["reply_preview_zone"]
        if zone["y_top"] <= y_position <= zone["y_bottom"]:
            return False
    if not text:
        return False
    
    text_stripped = text.strip()
    text_lower = text_stripped.lower()
    
    if text_stripped.startswith('@'):
        return False
    
    # Filter out system/action keywords (PaddleOCR approach)
    action_keywords = ['mentioned', 'replied', 'reacted', 'pinned', 'forwarded', 'shared', 'started', 'called', 'left', 'added', 'removed']
    if any(keyword in text_lower for keyword in action_keywords):
        return False
    
    # STRONGEST SIGNAL: Violet sender color (PaddleOCR uses this)
    if img is not None and bbox is not None:
        text_color = get_text_region_color(img, bbox)
        if text_color and is_violet_sender_color(text_color):
            return True
    
    # Check if text matches known aliases (PaddleOCR style)
    if text_lower in SENDER_ALIASES:
        if left_position < (image_width * 0.35):
            return True
    
    # Fuzzy match against aliases (more lenient)
    clean_for_match = text_lower.replace('ai', 'al').replace('0', 'o')
    matches = get_close_matches(clean_for_match, SENDER_ALIASES.keys(), n=1, cutoff=0.65)
    if matches:
        if left_position < (image_width * 0.35):
            return True
    
    # Pattern 1: All caps abbreviations (2-5 chars) - PaddleOCR approach
    if re.match(r'^[A-Z]{2,5}$', text_stripped):
        if left_position < (image_width * 0.35):
            return True
    
    # Pattern 2: Proper name format (Cap Followed By Lowercase, can have spaces)
    if re.match(r'^[A-Z][a-z]+(\s[A-Z][a-z]+)*$', text_stripped):
        if left_position < (image_width * 0.35) and text_lower not in REJECT_WORDS:
            return True
    
    # Pattern 3: Mixed case with dashes (like "Jo-Ann P.M.")
    if re.match(r'^[A-Z][a-zA-Z]*(-[A-Z][a-zA-Z]*)*(\s[A-Z][a-zA-Z.]*)*$', text_stripped):
        if left_position < (image_width * 0.35) and text_lower not in REJECT_WORDS:
            return True
    
    return False

def process_viber_chat_v5(folder_path):
    image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg'))])
    all_results = []
    
    global_state = {
        "last_active_sender": "Unknown",
        "sender_palette": {},
        "sender_message_history": {},
    }

    state = {
        "active_sender": "Unknown",
        "active_timestamp": None,
        "text_buffer": [],
        "conf_buffer": [],
        "last_y": 0,
        "last_sender": "Unknown",
        "last_left_x": None,
        "pending_reply_data": None,
        "reply_preview_zone": None,
        "sender_message_history": {},
    }

    def get_current_message(screenshot_filename=None, reply_to=None):
        if not state["text_buffer"]:
            return None
        message_body = " ".join(state["text_buffer"]).strip()
        message_body = re.sub( r'\b([01]?\d|2[0-3])[:.]([0-5]\d)\s*(AM|PM|am|pm)?\b', '', message_body ) if not message_body.strip().isdigit() else message_body
        message_body = message_body.replace(';', ',')
        message_body = message_body.replace(': ', ', ')
        message_body = re.sub(r'([0-2]?\d)\.O(\d)', r'\1.0\2', message_body)
        message_body = re.sub(r'([0-2]?\d)\sI(\d)', r'\1 1\2', message_body)
        message_body = re.sub(r'\s+', ' ', message_body)
        if not message_body:
            return None
        now_date = datetime.now().strftime("%Y-%m-%d")
        message_obj = {
            "timestamp": f"{now_date} {state['active_timestamp']}" if state["active_timestamp"] else None,
            "sender_raw": state["active_sender"],
            "sender": resolve_sender_alias(state["active_sender"]),
            "message": message_body,
            "message_type": "reply" if reply_to else "text",
            "confidence_score": round(float(np.mean(state["conf_buffer"])), 2)
        }
        if reply_to and reply_to.get("message_preview"):
            message_obj["reply_to"] = reply_to
            if reply_to.get("sender") or (reply_to.get("message_preview") and len(reply_to.get("message_preview", "")) > 5):
                message_obj["message_type"] = "reply"
            state["pending_reply_data"] = None
            if message_body.strip().isdigit():
                 message_obj["message_type"] = "numeric_reply"
        if screenshot_filename:
            message_obj["screenshot_path"] = screenshot_filename
        return message_obj

    for img_file in image_files:
        img = cv2.imread(os.path.join(folder_path, img_file))
        if img is None: continue
        h, w, _ = img.shape
        
        state["text_buffer"] = []
        state["conf_buffer"] = []
        state["last_y"] = 0
        state["last_left_x"] = None
        state["active_sender"] = global_state["last_active_sender"]
        state["current_timestamp"] = None
        state["active_timestamp"] = None
        bubbles = detect_bubbles(img)
        results = []
        bubble_objects = []

        for x, y, bw, bh in bubbles:
            bubble_img = img[y:y+bh, x:x+bw]
            ocr_results = reader.readtext(bubble_img)

            bubble_objects.append({
                "x": x,
                "y": y,
                "w": bw,
                "h": bh,
                "ocr": ocr_results,
                "is_image": is_image_bubble(bubble_img, ocr_results)
            })


            results.sort(key=lambda x: x[0][0][1])
        file_messages = []
        first_element = True
        bubble_idx = 0

        while bubble_idx < len(bubble_objects):
            bubble = bubble_objects[bubble_idx]
            x, y, w, h = bubble["x"], bubble["y"], bubble["w"], bubble["h"]
            ocr_results = bubble["ocr"]
            is_image = bubble["is_image"]

            # TRY TO DETECT REPLY BLOCK FIRST
            reply_block_result, next_idx = detect_reply_block(bubble_objects, bubble_idx)
            
            
            if reply_block_result:
                # We found a reply block pattern
                reply_msg = {
                    "timestamp": None,
                    "sender_raw": reply_block_result["reply_sender"],
                    "sender": reply_block_result["reply_sender"],
                    "message": reply_block_result["reply_text"],
                    "message_type": "reply",
                    "confidence_score": reply_block_result["confidence"],
                    "screenshot_path": img_file,
                    "reply_to": reply_block_result["reply_data"]
                }
                file_messages.append(reply_msg)
                state["active_sender"] = reply_block_result["reply_sender"]
                global_state["last_active_sender"] = state["active_sender"]
                bubble_idx = next_idx
                first_element = False
                continue

            if is_image:
                msg = {
                    "timestamp": None,
                    "sender_raw": state["active_sender"],
                    "sender": resolve_sender_alias(state["active_sender"]),
                    "message": "photo image",
                    "message_type": "image",
                    "confidence_score": 0.7,
                    "screenshot_path": img_file
                }

                file_messages.append(msg)

                state["pending_reply_data"] = {
                    "message_preview": "Photo message",
                    "sender": msg["sender"]
                }
                bubble_idx += 1
                continue

            for (bbox, raw_text, conf) in ocr_results:
                    raw_text = raw_text.strip()
                    if conf < OCR_CONF_THRESHOLD and not raw_text.isdigit():
                        continue

                    raw_text = raw_text.strip()
                    if not raw_text:
                        continue

                    x_coords = [p[0] for p in bbox]
                    y_coords = [p[1] for p in bbox]

                    left_x = min(x_coords) + x
                    y_top = min(y_coords) + y
                    y_bottom = max(y_coords) + y

                    if is_viber_date_separator(raw_text, left_x, w, img, bbox):
                        print(f"[FILTERED DATE SEPARATOR] {raw_text[:50]}")
                        continue

                    t_fixed = re.sub(r'(?<=\d)O|O(?=\d)', '0', raw_text.upper())

                    time_match = re.search(r'\b([0-2]?\d)[:.\s]*([0-5]\d)\s*(AM|PM)?', t_fixed)
                    found_ts = None
                    body_text = raw_text
                    if time_match:
                        found_ts = f"{time_match.group(1).zfill(2)}:{time_match.group(2)} {time_match.group(3) or ''}".strip()
                        body_text = re.sub(r'\b([0-2]?\d)[:.\s]*([0-5]\d)\s*(AM|PM)?\b', '', raw_text, flags=re.I).strip()


                    is_sender = is_valid_sender(raw_text, left_x, w, y_top, img, bbox,
                                    is_first_in_screenshot=first_element, state=state)
                    print(f"Bubble position: left_x={left_x}, bubble_side={'LEFT' if left_x < w * 0.4 else 'RIGHT'}, text={raw_text[:50]}")
                    if is_sender:

                        msg = get_current_message(img_file)
                        if msg:
                            file_messages.append(msg)

                        state["active_sender"] = raw_text.strip()
                        global_state["last_active_sender"] = state["active_sender"]

                        state["text_buffer"] = []
                        state["conf_buffer"] = []
                        state["last_y"] = y_bottom
                        first_element = False
                        continue

                    if found_ts and left_x > w * 0.6:
                        state["active_timestamp"] = found_ts
                        state["current_timestamp"] = found_ts
                        continue

                    reply_data = is_reply_to_previous(body_text, file_messages, current_sender=state["active_sender"])
                    if reply_data:
                        state["pending_reply_data"] = reply_data

                    clean_body = body_text.strip()
                    if not clean_body:
                        continue
                        file_messages[-1]["message"] += " " + clean_body
                        file_messages[-1]["confidence_score"] = min(file_messages[-1]["confidence_score"], conf)
                    else:
        
                        ts = state['active_timestamp']
                        if found_ts:
                            ts = found_ts
                            state['active_timestamp'] = ts

                        msg_type = "text"
                        reply_to_data = state["pending_reply_data"]
                        

                        if '@' in clean_body or reply_to_data:
                            msg_type = "reply"
                            if '@' in clean_body and not reply_to_data:
                                reply_to_data = is_reply_to_previous(clean_body, file_messages, current_sender=state["active_sender"], threshold=0.4)
                        

                        if file_messages and len(file_messages) > 0:
                            last_msg = file_messages[-1]
                            if last_msg.get("sender") != resolve_sender_alias(state["active_sender"]):
    
                                if not reply_to_data and 'replying' in clean_body.lower():
                                    reply_to_data = is_reply_to_previous(clean_body, file_messages, current_sender=state["active_sender"])
                                    msg_type = "reply"
                        
                        new_msg = {
                            "timestamp": f"{datetime.now().strftime('%Y-%m-%d')} {ts}" if ts else None,
                            "sender_raw": state["active_sender"],
                            "sender": resolve_sender_alias(state["active_sender"]),
                            "message": clean_body,
                            "message_type": msg_type,
                            "confidence_score": conf,
                            "screenshot_path": img_file
                        }
                        

                        if reply_to_data:
                            new_msg["reply_to"] = reply_to_data
                        
                        file_messages.append(new_msg)
                        state["pending_reply_data"] = None

                    state["last_y"] = y_bottom

                    if state["last_left_x"] is None:
                                state["last_left_x"] = left_x

                    first_element = False
            
            bubble_idx += 1
    

   
        last_msg = get_current_message(img_file, reply_to=state["pending_reply_data"])
        if last_msg:
            file_messages.append(last_msg)
            state["text_buffer"] = []
            state["conf_buffer"] = []
            state["pending_reply_data"] = None  
        all_results.append({
            "file": img_file,
            "messages": file_messages
        })
            
    for file_result in all_results:
        file_result["messages"].sort(key=lambda m: m.get("timestamp") or "")

    return all_results



def merge_message_fragments(messages):

    if not messages:
        return messages
    
    noise_tokens = {'am', 'pm', 'edited', 'forwarded', 'pin', 'pinned', 'copied', 'copy'}
    
    def get_timestamp_key(ts):
        if ts is None:
            return "__NONE__" 
        try:
            dt = datetime.strptime(ts, "%Y-%m-%d %H:%M")
            return dt
        except:
            return "__ERROR__"
    
    def is_noise_only(text):
        """Check if text is just noise tokens."""
        words = text.lower().split()
        return all(w in noise_tokens for w in words) if words else False
    
    merged = []
    current_group = None
    
    for msg in messages:

                # Only images should force a hard break
        if msg.get("message_type") == "image":
            if current_group:
                merged.append(current_group)
                current_group = None
            merged.append(msg)
            continue

        can_merge = False
        if current_group is not None:
            can_merge = (
                msg.get("sender") == current_group.get("sender") and
                msg.get("screenshot_path") == current_group.get("screenshot_path") and
                msg.get("message_type") == current_group.get("message_type")
            )
            
            if can_merge:
                current_ts_key = get_timestamp_key(current_group.get("timestamp"))
                msg_ts_key = get_timestamp_key(msg.get("timestamp"))

                if current_ts_key != msg_ts_key:
                    can_merge = False
        
        if can_merge:

            msg_text = msg.get("message", "").strip()
            if not is_noise_only(msg_text) and len(msg_text) > 0:
                current_group["message"] = (current_group.get("message", "") + " " + msg_text).strip()
                current_group["confidence_score"] = min(
                    current_group.get("confidence_score", 1.0),
                    msg.get("confidence_score", 1.0)
                )
        else:
            if current_group:
                merged.append(current_group)
            current_group = msg.copy()
    
    if current_group:
        merged.append(current_group)
    
    return merged


if __name__ == "__main__":
    final_data = process_viber_chat_v5(IMAGE_FOLDER)

    for file_result in final_data:
        file_result["messages"] = merge_message_fragments(file_result["messages"])
    
    previous_sender = None

    for file_result in final_data:
        for i, message in enumerate(file_result["messages"]):
            file_result["messages"][i], previous_sender = post_process_message(message, previous_sender)

        
        with open("viber_fixed_files.json", "w", encoding="utf-8") as f:
            json.dump(final_data, f, indent=2, ensure_ascii=False)
        print("Done! Post-processing applied. Results saved to viber_fixed_files.json")


