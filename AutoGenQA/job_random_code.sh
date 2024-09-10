#!/bin/sh
#$ -cwd
#$ -l node_h=1
#$ -l h_rt=24:00:00
cd /gs/fs/tga-hatakeyama/
cd code/AutoGenQA/
module load cuda
module load miniconda/24.1.2
eval "$(/apps/t4/rhel9/free/miniconda/24.1.2/bin/conda shell.bash hook)"
conda activate gen
python RandomCodeTokkun.py
