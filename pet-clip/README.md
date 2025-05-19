## Install

```Shell
conda env create -f env.yml 
conda activate pet-clip

# Navigate to the 'transformer_maskgit' directory and install the required packages
cd transformer_maskgit
pip install -e .

# Return to the root directory
cd ..

# Navigate to the 'CT_CLIP' directory and install its required packages
cd CT_CLIP
pip install -e .

# Return to the root directory
cd ..
```

## Train 

```Shell
Prepair data format as follow : 

root_directory/
└── PETCT 2017/
    ├── THANG 1/
    │   ├── images/
    │   │   ├── abdomen_pelvis/
    │   │   │   ├── img1.npy
    │   │   │   └── ...
    │   │   ├── chest/
    │   │   └── head_neck/
    │   └── reports/
    │       ├── abdomen_pelvis/
    │       │   ├── img1.txt
    │       │   └── ...
    │       ├── chest/
    │       └── head_neck/
    └── ...
```

### Configure Training Script

In the scripts/run_train_ctvit.py file, update the following code segment with your data paths:

``` python

trainer = CTClipTrainer(
    clip,
    root='path_to_PET_report_paired_data',
    comparation_path='path_to_comparation_data', # path to comparation data if you want to use comparation data else None
    batch_size = 8,
    tokenizer=tokenizer,
    results_folder=f"results/{experiment_name}", # path to save results
    num_train_steps = 100001,
    num_workers = 8,
    logger=logger,
)

```

### Start Training



Training script: 
```bash
sh script/pet-clip.sh 
```