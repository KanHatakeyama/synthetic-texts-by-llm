#!/bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=12:00:00
cd /gs/fs/tga-hatakeyama/
cd code/AutoGenQA/
module load cuda
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate gen
python DirctGenQAFromText.py --model_path ../../model/Mixtral-8x22B-Instruct-v0.1.Q5_K_M-00001-of-00004.gguf --ds_name hpprc/wikipedia-20240101


