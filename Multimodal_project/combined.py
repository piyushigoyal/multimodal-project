import json
import random

def sample_jsonl(file_path, num_samples):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    return random.sample(lines, num_samples)

# Paths to your JSONL files
file1_path = 'gen_qa_outputs/text/tictactoe.jsonl'
file2_path = 'gen_qa_outputs/text_img/tictactoe.jsonl'
output_path = 'gen_qa_outputs/combined_tictactoe.jsonl'

# Sample 5 lines from each file
samples_file1 = sample_jsonl(file1_path, 5)
samples_file2 = sample_jsonl(file2_path, 5)

# Combine and write to new file
with open(output_path, 'w', encoding='utf-8') as out_file:
    for line in samples_file1 + samples_file2:
        out_file.write(line)

print(f"Sampled 5 lines from each file and wrote to {output_path}")