
## Install

```Shell
conda env create -f env.yml 
conda activate saturn
```

## Train

Data format example: ./data_example/data.json


### Pretrain (feature alignment)


Training script with DeepSpeed ZeRO-2: [`bash/align.sh`].


### Visual Instruction Tuning


Training script with DeepSpeed ZeRO-2: [`bash/finetune.sh`].


### Inference 

After training, place the LoRA and projector weight files in the paths specified by the parameters --lora_path and --pretrain_mm_mlp_adapter (and any other required parameters) in the [`bash/infer.sh`] script and run: 

```bash
sh bash/infer.sh
```

This code is based on previous work: https://github.com/haotian-liu/LLaVA
