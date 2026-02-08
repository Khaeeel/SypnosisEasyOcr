import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" 

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

print("Loading model (this may take a moment)...")
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="cpu"
)

print("Creating pipeline...")
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
    temperature=0.2,
    top_p=0.9
)

# Load data from viber_fixed_files.json
data_file = "viber_fixed_files.json"
if not os.path.exists(data_file):
    print(f"Error: {data_file} not found!")
    exit(1)

print(f"Loading data from {data_file}...")
with open(data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

def flatten_conversation(data):
    """Extract all messages from viber data structure"""
    lines = []
    
    for file_entry in data:
        messages = file_entry.get("messages", [])
        for item in messages:
            msg = item.get("message", "").strip()
            sender = item.get("sender", "Unknown")
            timestamp = item.get("timestamp", "")
            
            # Skip short/empty messages and noise
            if len(msg) < 3:
                continue
            
            lines.append(f"[{timestamp}] {sender}: {msg}")
    
    return "\n".join(lines)


conversation_text = flatten_conversation(data)

print(f"Extracted {len(conversation_text.split(chr(10)))} message lines")

prompt = f"""You are an assistant that analyzes team discussions.

Tasks:
1. Write a concise meeting summary (5–7 sentences).
2. Extract clear action items with owner and task.
3. Identify the main objective.
4. List important decisions or agreements.

Rules:
- Ignore greetings, duplicates, OCR noise, and filler
- Be factual and concise
- Do not invent information

Transcript:
{conversation_text}
"""

print("\nGenerating analysis...")
result = generator(prompt)[0]["generated_text"]

print("\n================ MODEL OUTPUT ================\n")
print(result)

# Evaluation
expected_keywords = [
    "task management",
    "demo",
    "persistence",
    "undo",
    "status",
    "due today",
    "upcoming",
    "done",
    "task history",
    "objective"
]

found_keywords = [k for k in expected_keywords if k.lower() in result.lower()]

expected_actions = {
    "Bharon": ["backend", "api", "status", "persistence"],
    "Dominic": ["frontend", "state", "group"],
    "Lebron": ["validation", "refresh", "undo"],
    "Balonda": ["layout", "labels"]
}

matched_owners = []
for owner, keywords in expected_actions.items():
    if owner.lower() in result.lower():
        for k in keywords:
            if k.lower() in result.lower():
                matched_owners.append(owner)
                break

noise_terms = ["hello", "unknown", "screenshot"]
noise_found = [n for n in noise_terms if n.lower() in result.lower()]

print("\n================ EVALUATION =================\n")
print(f"Keyword coverage: {len(found_keywords)} / {len(expected_keywords)}")
print("Matched keywords:", found_keywords)

print(f"\nAction items matched: {len(set(matched_owners))} / {len(expected_actions)}")
print("Matched owners:", list(set(matched_owners)))

print("\nNoise found:", noise_found if noise_found else "None")

print("\n================ VERDICT =================\n")
if len(found_keywords) >= 7 and len(set(matched_owners)) >= 3 and not noise_found:
    print("✅ MODEL FIT: Very good fit for your JSON")
elif len(found_keywords) >= 5:
    print("⚠️ MODEL FIT: Usable, but prompt tuning recommended")
else:
    print("❌ MODEL FIT: Weak extraction")