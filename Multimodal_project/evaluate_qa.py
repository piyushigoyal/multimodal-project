import json
import re

def extract_answer(generated_text):
    """Extracts answer choice (A-D) after 'assistant' in generated text."""
    match = re.search(r'assistant\s+([A-D])', generated_text)
    return match.group(1) if match else None

def compute_accuracy(samples):
    total = 0
    correct = 0

    for sample in samples:
        gt = sample.get("ground_truth")
        pred = extract_answer(sample.get("generated_answer", ""))

        if pred is not None:
            total += 1
            if pred == gt:
                correct += 1
        else:
            print(f"Warning: Could not extract answer from sample with question: {sample.get('question')}")

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nAccuracy: {accuracy * 100:.2f}% ({correct}/{total})")
    return accuracy

def load_samples(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        output_data = [json.loads(line) for line in f]
        return output_data

if __name__ == "__main__":
    # Replace this with your actual JSON file path
    json_file_path = "gen_qa_outputs/text/chess.jsonl"

    samples = load_samples(json_file_path)
    compute_accuracy(samples)