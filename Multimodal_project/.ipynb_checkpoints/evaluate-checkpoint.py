import json
import re
import numpy as np
import editdistance


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

def accuracy(preds, truths):
    """
    Calculates exact match accuracy.
    """
    print(len(preds))
    correct = sum([p == t for p, t in zip(preds, truths)])
    return correct / len(preds) if preds else 0

    
# input_path = "mathvista_data/testmini/data.jsonl"
output_path = "outputs/chameleon_outputs.jsonl"

# with open(input_path, "r", encoding="utf-8") as f:
#     input_data = [json.loads(line) for line in f]

with open(output_path, "r", encoding="utf-8") as f:
    output_data = [json.loads(line) for line in f]

preds = []
truths = []
misc = []

for sample in output_data:
    pid = sample["pid"]
    query = sample["query"]
    # model_output = sample.get("revised_extracted_answer")
    model_output = sample.get("prediction")
    if model_output is None:
        model_output = sample["extracted_answer"]
    # model_output = sample["generated_answer"]
    # ground_truth = sample["ground_truth_answer"]
    ground_truth = sample["answer"]
    handled = False
    if "Choices" in query:
        options = get_choices(output_data, pid)
        # print(options)
        idx = select_mc_option(model_output, options)
        # print(idx)
        extracted_answer = options[idx] if 0 <= idx < len(options) else ""
        # print(extracted_answer)

        # Save extracted prediction to the sample
        # sample["prediction"] = extracted_answer
        sample["final_pred"] = extracted_answer
        handled = True
        # sample["pred_index"] = idx
    
    # if not handled and is_numeric_only(model_output):
    #     extracted_answer = model_output.strip()
    #     sample["prediction"] = extracted_answer
    #     handled = True
    
    # # Append only if extracted_answer was determined
    # if handled:
    #     preds.append(sample["prediction"])
    #     truths.append(ground_truth)
    # else:
    #     misc.append(sample)
    
    elif is_numeric_only(model_output):
        # sample["prediction"] = model_output.strip()
        sample["final_pred"] = model_output.strip()
        handled = True

    # NEW: default handling for other cases
    if not handled:
        # sample["prediction"] = model_output.strip()
        sample["final_pred"] = model_output.strip()

    # Now always append
    # preds.append(sample["prediction"])
    preds.append(sample["final_pred"])
    truths.append(ground_truth)

acc = accuracy(preds, truths)
print(f"Accuracy: {acc:.2%}")
# print(preds)
# print(truths)
# output_2 = "mathvista_data/testmini/cot_acc.jsonl"
# # Save enriched dataset
# with open(output_2, "w", encoding="utf-8") as f:
#     for item in output_data:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")