import json

def calculate_accuracy(jsonl_path):
    """
    Calculates accuracy of predictions in a .jsonl file
    comparing extracted_answer with ground_truth_answer.

    Args:
        jsonl_path (str): Path to the JSONL file.

    Returns:
        float: Accuracy (correct / total)
        int: Number of correct predictions
        int: Total number of predictions considered
    """
    correct = 0
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            pred = item.get("extracted_answer")
            true = item.get("ground_truth_answer")

            if pred is not None and true is not None:
                # Strip spaces and compare as strings
                if str(pred).strip() == str(true).strip():
                    correct += 1
                total += 1

    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {accuracy:.2%} ({correct}/{total})")
    return accuracy, correct, total

print(calculate_accuracy("mathvista_data/testmini/master_output.jsonl"))