# Install environment

```bash
conda env create --file cosmos-predict1.yaml
conda activate cosmos-predict1
pip install -r requirements.txt
# Install the dependencies.
pip install -r requirements.txt
# Patch Transformer engine linking issues in conda environments.
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10
# Install Transformer engine.
pip install transformer-engine[pytorch]==1.12.0
# Install Apex for full training with bfloat16.
git clone https://github.com/NVIDIA/apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./apex
```

# Run the code

### Prepare the data
Create a train.txt and val.txt file in the finetune folder
Each line in the file is the path to the image

### Run the code
copy path of train.txt and val.txt to config.yaml
```bash
python finetune/training_custom_new.py
```



