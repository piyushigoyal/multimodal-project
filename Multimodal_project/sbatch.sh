#!/bin/bash
#SBATCH --job-name=txtimg_chess
#SBATCH --mem-per-cpu=60000
#SBATCH -n 8
#SBATCH --time=7000
#SBATCH --gpus=1

module load stack/.2024-06-silent
module load gcc/12.2.0
source ~/.bashrc
conda activate myenv

python3 sample_qa.py \
  --input gen_qa_set/chess.jsonl \
  --img_dir LVLM-Playground/benchmark/perceive/chess \
  --output gen_qa_outputs/text_img/chess.jsonl \
  --model_path ../../../../home/pgoyal/qwen-vl-7B