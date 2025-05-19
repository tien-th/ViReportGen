
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer
from log import Logger

# pretrain = 'microsoft/BiomedVLP-CXR-BERT-specialized'


# text_encoder = BertModel.from_pretrained(pretrain)



pretrain = 'vinai/phobert-base'
tokenizer = BertTokenizer.from_pretrained(pretrain, do_lower_case=True)
text_encoder = BertModel.from_pretrained(pretrain)

from ctvit import CTViT

image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 20,
    temporal_patch_size = 10,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)
#dim_image = 131072,


clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_text = 768,
    dim_image = 294912,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False
)

# save ckpt of CT-CLIP
import torch
# torch.save(clip.state_dict(), "/home/user01/aiotlab/htien/pet-clip/scripts/models/CT-CLIP-Related/test.pt")
# load ckpt of CT-CLIP 
# clip.load_state_dict(torch.load("/home/jovyan/shared/tienhuu060102/data-petct/pretrained_weights/CT-CLIP_v2_PhoBert.pt"))
from collections import OrderedDict

# Giả sử bạn đã load state_dict như sau:
checkpoint_path = '/home/jovyan/workspace/admin/pet-clip/results/CTVit_CLIP_Training_Experiment_0/CTClip.32000.pt'
state_dict = torch.load(checkpoint_path)

# Tạo một OrderedDict mới để loại bỏ tiền tố "module."
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # Nếu khóa bắt đầu bằng "module.", ta bỏ đi đoạn đó
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v

# Cuối cùng load lại vào mô hình của bạn
clip.load_state_dict(new_state_dict)

# clip.text_transformer = text_encoder 

experiment_name = "CTVit_CLIP_Training_Experiment"
logger = Logger(experiment_name=experiment_name)

trainer = CTClipTrainer(
    clip,
    root='/home/jovyan/shared/tienhuu060102/data-petct/PET_report_paired_fixed',
    batch_size = 12,
    tokenizer=tokenizer,
    results_folder=f"results/{experiment_name}",
    num_train_steps = 100001,
    num_workers = 2,
    logger=logger,
)

trainer.train()
