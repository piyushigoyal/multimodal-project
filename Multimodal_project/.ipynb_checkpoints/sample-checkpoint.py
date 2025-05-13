import json
import os
from PIL import Image
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
import re
import editdistance
import numpy as np

# Paths
base_dir = "mathvista_data/testmini"
# image_dir = os.path.join(base_dir, "images")
jsonl_path = os.path.join(base_dir, "data.jsonl")
# jsonl_path = "mathvista_data/testmini/misc_samples.jsonl"
output_file = os.path.join(base_dir, "prompt_outputs.jsonl")
model_path = "../../../../work/sachan/piyushi/models/qwen-vl-3B-it"

processor = AutoProcessor.from_pretrained(model_path, max_pixels=1280*28*28, trust_remote_code=True)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    local_files_only=True,
    device_map="auto",
    torch_dtype="auto"
)

# Load dataset
with open(jsonl_path, "r", encoding="utf-8") as f:
    data = [json.loads(line) for line in f]

results = []

def clean_pred(pred, options=[]):
    if pred in options: 
        return pred  # skip if clean enough
    answer_inds = ['Answer:', '*Answer*:', '*Answer:*', '**Answer:**', '**Answer**:']
    pred = pred.strip()
    tmp_ind = ''
    for answer_ind in answer_inds:
        if answer_ind in pred:
            tmp_ind = answer_ind
            break
    if len(tmp_ind):
        pred = pred.split(tmp_ind)[-1].strip()
    else:
        pred = " ".join(pred.strip().split())
    return pred

def is_numeric_only(answer):
    """
    Returns True if the answer is a numeric value only: integer, float, or negative,
    possibly with whitespace around it.
    """

    return bool(re.fullmatch(r"\s*-?\d+(\.\d+)?\s*", answer))

def select_mc_option(target, options):
    target = clean_pred(target, options)
    # if model output the answer directly
    if target in options:
        return options.index(target)
    # if model output contain one unique option
    tmp_count = 0
    for i in range(len(options)):
        op = options[i]
        if op in target:
            tmp_count += 1
            tmp_idx = i
    if tmp_count == 1:
        return tmp_idx
    # if model output a character, use it as index of available choices
    sequential_characters = [chr(ord('A') + i) for i in range(len(options))]
    if target in sequential_characters:  
        return sequential_characters.index(target)
    # if all failed, select the most similar option
    target = target.lower().strip()
    n = len(options)
    options = [x.lower().strip() for x in options]
    for ix, option in enumerate(options):
        if option == target:
            return ix
    contains = []
    for ix, option in enumerate(options):
        if target in option:
            contains.append(ix)
    if len(contains) == 1:
        return contains[0]
    distances = [editdistance.eval(opt, target) for opt in options]
    return np.argmin(distances)

def get_choices(dataset, pid):
    """
    Retrieves the options for the given pid

    Args:
        dataset (list[dict]): The dataset containing samples with pid, query, etc.
        pid (str): The target pid to find.

    Returns:
        int: Index of the selected option.
    """
    
    for sample in dataset:
        if sample.get("pid") == pid:
            options = sample["choices"]
            return options
    raise ValueError(f"PID {pid} not found in dataset.")

def resize_if_needed(image, min_size=28):
    width, height = image.size
    if height < min_size or width < min_size:
        new_height = max(height, min_size)
        new_width = max(width, min_size)
        image = image.resize((new_width, new_height), Image.BILINEAR)
    return image

#--------------1 round of prompting--------------#

# def inference_batch(image_paths, prompts, model, processor, max_new_tokens=256):
#     """
#     Perform batch inference for a list of image paths and corresponding prompts.

#     Args:
#       image_paths: List of image file paths.
#       prompts: List of text prompts corresponding to each image.
#       model: Loaded VLM model.
#       processor: Loaded processor.
#       max_new_tokens:

#     Returns:
#       List of model-generated output texts for each image-text pair in the batch.
#     """
#     # Prepare the messages in the required format for batch inference
#     # messages_batch = [
#     #     {
#     #         "role": "user",
#     #         "content": [
#     #             {"type": "image", "image": image_path},
#     #             {"type": "text", "text": prompt},
#     #         ],
#     #     }
#     #     for image_path, prompt in zip(image_paths, prompts)
#     # ]
#     messages_batch = []
#     for img_path, prompt in zip(image_paths, prompts):
#         image = Image.open(img_path).convert("RGB")
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image", "image": image},
#                     {"type": "text", "text": prompt},
#                 ],
#             }]
#         messages_batch.append(messages)

#     # print("\n--- DEBUG CHAT TEMPLATE ---")
#     # for idx, messages in enumerate(messages_batch):
#     #     user_contents = [c['text'] for m in messages for c in m['content'] if c['type'] == 'text']
#     #     print(f"[{idx}] user text:", user_contents)
#     #     chat_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#     #     print(f"[{idx}] chat_prompt:\n", chat_prompt)

#     # Prepare the input for the processor
#     texts = [
#         processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#         for messages in messages_batch
#     ]

#     # image_inputs = [Image.open(msg["content"][0]["image"]).convert("RGB") for msg in messages_batch]
#     image_inputs, video_inputs = process_vision_info(messages_batch)
#     inputs = processor(
#         text=texts[0],
#         images=image_inputs[0],
#         videos=video_inputs,
#         padding=True,
#         return_tensors="pt",
#     )

#     inputs = inputs.to("cuda")
    
#     # Perform batch inference for all images
#     with torch.no_grad():
#         generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
#     generated_ids_trimmed = [
#         out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
#     ]
#     output_texts = processor.batch_decode(
#         generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#     )
#     # del inputs, output
#     torch.cuda.empty_cache()
#     return output_texts

#--------------2-round prompting--------------#


def inference_batch(image_paths, prompts, model, processor, max_new_tokens=256):
    results = []
    
    system_prompt = """You are an AI assistant that solves questions by referring to the associated images. Analyze the content, context, and notable features of the images. Provide an answer that covers the important aspects of the image."""
    
    for img_path, prompt in zip(image_paths, prompts):
        # image = Image.open(img_path).convert("RGB")
        try:
            image = Image.open(img_path).convert("RGB")
            image = resize_if_needed(image)
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            continue


        # First prompt: full image + question
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "image", "image": image},
        #             {"type": "text", "text": "Solve the given question using step-by-step reasoning." + prompt},
        #         ],
        #     }
        # ]
        # Recreate the message list fresh for each image
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Describe the image and then use that to solve the given question using step-by-step reasoning" + prompt},
                ],
            },
        ]

        # Stage 1: Get full model response
        text_1 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info([messages])
        inputs = processor(
            text=[text_1],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            gen_ids_1 = model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed_ids_1 = gen_ids_1[:, inputs.input_ids.shape[1]:]
        full_response = processor.batch_decode(trimmed_ids_1, skip_special_tokens=True)[0]

        torch.cuda.empty_cache()

        # Stage 2: Ask for only the integer answer
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": full_response}],
        })
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": "Given your previous response, extract the final numeric answer only."}],
        })

        text_2 = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text_2],
            images=[image],
            videos=None,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            gen_ids_2 = model.generate(**inputs, max_new_tokens=10)
        trimmed_ids_2 = gen_ids_2[:, inputs.input_ids.shape[1]:]
        final_answer = processor.batch_decode(trimmed_ids_2, skip_special_tokens=True)[0]

        torch.cuda.empty_cache()

        results.append({
            "generated_answer": full_response,
            "extracted_answer": final_answer
        })
        # break

    return results

def interactive_reprompt(image_path, prompt, model, processor, previous_response, max_new_tokens=256):
    # image = Image.open(image_path).convert("RGB")
    try:
        image = Image.open(image_path).convert("RGB")
        image = resize_if_needed(image)
    except Exception as e:
        print(f"Skipping {img_path} due to error: {e}")
        return None

    # Message history leading up to the revision
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": previous_response}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": "That doesn't seem correct. Observe the image carefully, think step-by-step and then provide the answer."}],
        },
    ]

    # Step 1: Get revised explanation
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed_ids = gen_ids[:, inputs.input_ids.shape[1]:]
    revised_response = processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0]

    torch.cuda.empty_cache()

    # Step 2: Extract numeric answer from revised response
    messages.append({
        "role": "assistant",
        "content": [{"type": "text", "text": revised_response}],
    })
    messages.append({
        "role": "user",
        "content": [{"type": "text", "text": "Now, extract only the final numeric answer from your revised response."}],
    })

    text_final = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs_final = processor(
        text=[text_final],
        images=[image],
        videos=None,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        gen_ids_final = model.generate(**inputs_final, max_new_tokens=10)
    trimmed_ids_final = gen_ids_final[:, inputs_final.input_ids.shape[1]:]
    revised_extracted_answer = processor.batch_decode(trimmed_ids_final, skip_special_tokens=True)[0]

    torch.cuda.empty_cache()

    return {
        "revised_generated_answer": revised_response,
        "revised_extracted_answer": revised_extracted_answer,
    }

# Inference loop with batching
batch_size = 1
results = []
# len(data)
preds, truths = [], []

progress_bar = tqdm(range(0, len(data), batch_size))
for i in progress_bar:
    batch = data[i:i+batch_size]

    image_paths, prompts, metadata = [], [], []
    for item in batch:
        image_rel_path = item["image_path"]
        # full_path = os.path.join(base_dir, os.path.basename(image_rel_path))

        if not os.path.exists(image_rel_path):
            print(f"Warning: Image not found at {image_rel_path}, skipping.")
            continue

        image_paths.append(image_rel_path)
        prompts.append(item["query"])
        # metadata.append({
        #     "pid": item["pid"],
        #     "image_path": image_rel_path,
        #     "query": item["query"],
        #     "ground_truth_answer": item["answer"]
        # })
        metadata.append(item)

    if not image_paths:
        continue

    # Run model
    answers = inference_batch(image_paths, prompts, model, processor)
    
    # Update original items
    for idx, (item, result) in enumerate(zip(metadata, answers)):
        item["generated_answer"] = result["generated_answer"]
        item["extracted_answer"] = result["extracted_answer"]
        # print(item)
        # Interactive retry if answer is incorrect
        # if item["answer"] != result["extracted_answer"]:
        #     retry_result = interactive_reprompt(
        #         image_path=image_paths[idx],
        #         prompt=prompts[idx],
        #         model=model,
        #         processor=processor,
        #         previous_response=result["generated_answer"]
        #     )
        #     if retry_result is None:
        #         continue
        #     item.update(retry_result)
        # --- Post-process answer for accuracy evaluation ---
        model_output = result["extracted_answer"]
        query = item["query"]
        ground_truth = item["answer"]

        handled = False

        if "Choices" in query:
            options = get_choices(data, item["pid"])
            choice_idx = select_mc_option(model_output, options)
            prediction = options[choice_idx] if 0 <= choice_idx < len(options) else ""
            handled = True

        elif is_numeric_only(model_output):
            prediction = model_output.strip()
            handled = True

        if not handled:
            prediction = model_output.strip()

        item["prediction"] = prediction
        preds.append(prediction)
        truths.append(ground_truth)

    # Compute and update accuracy in progress bar
    if preds:  # avoid ZeroDivisionError
        acc = sum(p == t for p, t in zip(preds, truths)) / len(preds)
        progress_bar.set_postfix({"Accuracy": f"{acc:.2%}"})
    torch.cuda.empty_cache()
    # break

# Save results
# with open(output_file, "w", encoding="utf-8") as f:
#     for r in results:
#         f.write(json.dumps(r, ensure_ascii=False) + "\n")

# Overwrite the original file with updated entries
with open(output_file, "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        # break

print(f"Inference complete. Output saved to {output_file}")