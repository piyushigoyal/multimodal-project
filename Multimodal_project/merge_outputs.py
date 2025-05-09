import json

# Paths
subset_path = "mathvista_data/testmini/misc_samples.jsonl"         # enriched with predictions
master_path = "mathvista_data/testmini/qwen_vl_outputs2.jsonl"         # full dataset to update
output_path = "mathvista_data/testmini/master_output.jsonl" # merged output

# Load files
with open(subset_path, "r", encoding="utf-8") as f:
    subset_data = [json.loads(line) for line in f]

with open(master_path, "r", encoding="utf-8") as f:
    master_data = [json.loads(line) for line in f]

# Create a lookup from pid to extracted answer
answer_lookup = {item["pid"]: item.get("extracted_answer") for item in subset_data}

# Merge extracted answers into master
for item in master_data:
    pid = item["pid"]
    if pid in answer_lookup:
        item["extracted_answer"] = answer_lookup[pid]

# Save merged results
with open(output_path, "w", encoding="utf-8") as f:
    for item in master_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Merged output written to {output_path}")
