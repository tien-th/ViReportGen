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


def build_vision_tower(vision_tower_cfg, **kwargs):
    '''
    cosmos
    '''

    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    # vision_tower = getattr(vision_tower_cfg, 'vision_tower', None)
    if vision_tower is not None:
        is_absolute_path_exists = os.path.exists(vision_tower)
    
        if is_absolute_path_exists:
            print("Loading Vision Tower from: ", vision_tower)

            cosmos = torch.jit.load(vision_tower) 
            encoder = cosmos.encoder
            encoder = encoder.to(torch.bfloat16)
            
            for param in encoder.parameters():
                param.requires_grad = False

            encoder.eval() 
            encoder.is_loaded = True
            encoder.hidden_size = 16384
            return encoder
        
        return None
    
    return None 
    
