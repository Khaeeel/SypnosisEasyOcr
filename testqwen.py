import json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

with open("viber_fixed_files.json", "r", encoding="utf-8") as f:
    data = json.load(f)

conversation = ""
for block in data:
    for msg in block["messages"]:
        conversation += f'{msg["sender"]}: {msg["message"]}\n'

prompt = f"""
You are an AI analyst. Extract structured information from noisy OCR chat logs.

OUTPUT FORMAT:

## PROJECT THREADS
- Topic
  - Key discussion points

## ACTION ITEMS
| Assignee | Task | Status | Date | Due |

## KEY UPDATES
- Important events

## LINKS & RESOURCES
- URLs

Conversation:
{conversation}
"""

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=500, temperature=0.2)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
