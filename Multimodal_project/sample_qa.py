import json
import os
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info  # Ensure this is accessible
import re

def main(args):
    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=1280*28*28, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto"
    )

    # Load dataset
    # with open(args.input, "r", encoding="utf-8") as f:
    #     dataset = json.load(f)
    with open(args.input, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]


    results = []

    for sample in tqdm(dataset, desc="Processing"):
        image_path = os.path.join(args.img_dir, sample["file"])
        image = Image.open(image_path).convert("RGB")
        
        example_prompt = sample.get("example_qa", "").strip()
        question = sample["question"].strip()
        full_prompt = f"{example_prompt}\nQuestion: {question}"
        ground_truth = sample["answer"]
        
        # example_prompt = sample["gt"].get("example_qa", "").strip()
        # question = sample["gt"]["question"].strip()
        # full_prompt = f"{example_prompt}\nQuestion: {question}"
        # ground_truth = sample["gt"]["answer"]

        messages = [
            {"role": "user",
             "content": [
                 {"type": "image"},
                 {"type": "text", "text": full_prompt}
             ]}
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                use_cache=True,
                temperature=1.5,
                min_p=0.1
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        results.append({
            "file": sample["file"],
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_text
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} results to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LVLM inference")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--img_dir", type=str, required=True, help="Path to image directory")
    parser.add_argument("--output", type=str, required=True, help="Path to save output JSONL file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen-VL model directory")

    args = parser.parse_args()
    main(args)