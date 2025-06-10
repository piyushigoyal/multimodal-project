import json
import os
import argparse
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from qwen_vl_utils import process_vision_info
from prompts import task_prompts  # <- Import prompts

def format_matrix(matrix):
    return "\n".join([" ".join(map(str, row)) for row in matrix])

def main(args):
    # Load model and processor
    processor = AutoProcessor.from_pretrained(args.model_path, max_pixels=1280*28*28, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        local_files_only=True,
        device_map="auto",
        torch_dtype="auto"
    )

    task_prompt = task_prompts.get(args.task.lower())
    if not task_prompt:
        raise ValueError(f"Task prompt not found for: {args.task}")

    with open(args.input, "r", encoding="utf-8") as f:
        dataset = [json.loads(line) for line in f if line.strip()]

    results = []

    for sample in tqdm(dataset, desc="Processing"):
        question = sample["question"].strip()
        ground_truth = sample["answer"]
        matrix = sample.get("matrix")
        example_prompt = sample.get("example_qa", "").strip()

        if matrix is None:
            continue  # or raise an error

        matrix_str = format_matrix(matrix)
        full_prompt = f"{task_prompt}\nGame State:\n{matrix_str}\nExamples:\n{example_prompt}\nQuestion: {question}"

        messages = [
            {"role": "user",
             "content": [
                 {"type": "text", "text": full_prompt}
             ]}
        ]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt").to(model.device)

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
            "file": sample.get("file", ""),
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
    parser = argparse.ArgumentParser(description="Run LVLM inference without image input")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Path to save output JSONL file")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Qwen-VL model directory")
    parser.add_argument("--task", type=str, required=True, help="Task name (e.g., chess, sudoku)")

    args = parser.parse_args()
    main(args)