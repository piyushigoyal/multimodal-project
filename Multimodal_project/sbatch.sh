#!/bin/bash
#SBATCH --job-name=txt_t
#SBATCH --mem-per-cpu=50000
#SBATCH --time=200
#SBATCH --gpus=1
#SBATCH --gres=gpumem:31g

module load stack/.2024-06-silent
module load gcc/12.2.0
source ~/.bashrc
conda activate myenv

python3 sample_text.py \
  --input gen_qa_set/tictactoe.jsonl \
  --output gen_qa_outputs/text/tictactoe.jsonl \
  --model_path ../../../../home/pgoyal/qwen-vl-7B \
  --task tictactoe