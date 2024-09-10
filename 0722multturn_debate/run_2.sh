#!/bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=4:00:00
echo "start"
export HF_HOME="/gs/bs/tga-hatakeyama/hf_cache"
export CUDA_CACHE_PATH=/gs/bs/tga-hatakeyama/cuda_cache
cd /gs/bs/tga-hatakeyama/0722multturn_debate/
module load cuda
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate synthtext

export CUDA_VISIBLE_DEVICES=0
nohup python free_multiturn_with_memory.py &
export CUDA_VISIBLE_DEVICES=1
python free_multiturn_with_memory.py
