conda env create -f environment.yml
conda activate llava_thaind
mamba install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=12.1 -c pytorch -c nvidia
mamba install einops
pip install beartype
# mamba install nvidia/label/cuda-12.6.0::cuda-toolkit
