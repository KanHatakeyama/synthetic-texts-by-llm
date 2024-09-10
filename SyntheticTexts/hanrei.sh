#!/bin/sh
#$ -cwd
#$ -l gpu_1=1
#$ -l h_rt=24:00:00
export HF_HOME="/gs/bs/tga-hatakeyama/hf_cache"
export CUDA_CACHE_PATH=/gs/bs/tga-hatakeyama/cuda_cache
cd /gs/bs/tga-hatakeyama/SyntheticTexts/
module load cuda
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate synthtext
python 0609hanrei.py
