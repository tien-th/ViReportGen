# import os
# from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2


# def build_vision_tower(vision_tower_cfg, **kwargs):
#     vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
#     is_absolute_path_exists = os.path.exists(vision_tower)
#     use_s2 = getattr(vision_tower_cfg, 's2', False)
#     if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
#         if use_s2:
#             return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
#         else:
#             return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

#     raise ValueError(f'Unknown vision tower: {vision_tower}')
import torch
import os
from .clip_encoder import CLIPVisionTower, CLIPVisionTowerS2
from .ctvit import CTViT

def build_vision_tower(vision_tower_cfg, **kwargs):
    '''
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    use_s2 = getattr(vision_tower_cfg, 's2', False)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or "ShareGPT4V" in vision_tower:
        if use_s2:
            return CLIPVisionTowerS2(vision_tower, args=vision_tower_cfg, **kwargs)
        else:
            return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    
    raise ValueError(f'Unknown vision tower: {vision_tower}')
    '''
    
    cvit = CTViT(
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
    cvit.hidden_size = 294912

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # vision_tower = getattr(vision_tower_cfg, 'vision_tower', None)
    if vision_tower is not None:
        is_absolute_path_exists = os.path.exists(vision_tower)
    
        if is_absolute_path_exists:
            print("Loading Vision Tower from: ", vision_tower)
            checkpoint = torch.load(vision_tower, map_location='cpu')
            from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
            consume_prefix_in_state_dict_if_present(checkpoint, prefix="module.")
            cvit.load_state_dict(checkpoint) 
            cvit = cvit.to(torch.bfloat16)
            
            for param in cvit.parameters():
                param.requires_grad = False

            cvit.eval() 
            cvit.hidden_size = 294912
            
            cvit.is_loaded = True
        
            return cvit
        
        return None
    
    return None 
    
