

# Setup

~~~
conda create -n synthtext python=3.11 -y
conda activate synthtext

pip install vllm==0.4.2
pip install datasets==2.19.1
~~~

# Run
~~~
#1gpuでOK
conda activate synthtext
export CUDA_VISIBLE_DEVICES=0
python 0530autogen.py # wikipediaの場合
~~~