import json
import os
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
# from transformers import ChameleonProcessor, ChameleonForConditionalGeneration
import re
import editdistance
import numpy as np

# Paths
base_dir = "LVLM-Playground/benchmark/qa/chess"
# image_dir = os.path.join(base_dir, "images")
jsonl_path = os.path.join(base_dir, "data.jsonl")
annotations_path = os.path.join(base_dir, "annotation.json")
# jsonl_path = "mathvista_data/testmini/misc_samples.jsonl"
output_file = "outputs/qa_chess.jsonl"
model_path = "../../../../home/pgoyal/qwen-vl-7B"

processor = AutoProcessor.from_pretrained(model_path, max_pixels=1280*28*28, trust_remote_code=True)
model = AutoModelForImageTextToText.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto"
)

# processor = ChameleonProcessor.from_pretrained(model_path)
# model = ChameleonForConditionalGeneration.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cuda")

# Load dataset
with open(annotations_path, "r", encoding="utf-8") as f:
    dataset = json.load(f)

data = dataset["annotations"]  # contains list of samples

# Run inference and save results
results = []

for sample in tqdm(data, desc="Processing"):
    image_path = os.path.join(base_dir, sample["file"])
    image = Image.open(image_path).convert("RGB")
    
    example_prompt = sample["gt"].get("example_qa", "").strip()
    question = sample["gt"]["question"].strip()
    full_prompt = f"{example_prompt}\nQuestion: {question}"
    ground_truth = sample["gt"]["answer"]
    # print(full_prompt)
    # Process input
    messages = [
        {"role": "user", 
         "content": [
            {"type": "image"},
            {"type": "text", "text": full_prompt}
        ]}
    ]
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

    # inputs = processor(text=full_prompt, images=image, return_tensors="pt").to(model.device)

    # Generate output
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128, use_cache = True, temperature = 1.5, min_p = 0.1)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Store result
    results.append({
        "file": sample["file"],
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": generated_text
    })

# Save to output file
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    for item in results:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Saved {len(results)} results to {output_file}")