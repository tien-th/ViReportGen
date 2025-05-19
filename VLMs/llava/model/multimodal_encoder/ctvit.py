from pathlib import Path
import copy
import math
from functools import wraps

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.autograd import grad as torch_grad
from torchvision import transforms as T, utils


import torchvision

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

# from vector_quantize_pytorch import VectorQuantize


# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def divisible_by(numer, denom):
    return (numer % denom) == 0

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def remove_vgg(fn):
    @wraps(fn)
    def inner(self, *args, **kwargs):
        has_vgg = hasattr(self, 'vgg')
        if has_vgg:
            vgg = self.vgg
            delattr(self, 'vgg')

        out = fn(self, *args, **kwargs)

        if has_vgg:
            self.vgg = vgg

        return out
    return inner

def pair(val):
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret

def cast_tuple(val, l = 1):
    return val if isinstance(val, tuple) else (val,) * l

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    device=torch.device('cuda')
    gradients = torch_grad(
        outputs = output,
        inputs = images,
        grad_outputs = torch.ones(output.size(), device = device),
        create_graph = True,
        retain_graph = True,
        only_inputs = True
    )[0]

    gradients = rearrange(gradients, 'b ... -> b (...)')
    return weight * ((gradients.norm(2, dim = 1) - 1) ** 2).mean()

def l2norm(t):
    return F.normalize(t, dim = -1)

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def safe_div(numer, denom, eps = 1e-8):
    return numer / (denom + eps)

# gan losses

def hinge_discr_loss(fake, real):
    return (F.relu(1 + fake) + F.relu(1 - real)).mean()

def hinge_gen_loss(fake):
    return -fake.mean()

def bce_discr_loss(fake, real):
    return (-log(1 - torch.sigmoid(fake)) - log(torch.sigmoid(real))).mean()

def bce_gen_loss(fake):
    return -log(torch.sigmoid(fake)).mean()

def grad_layer_wrt_loss(loss, layer):
    return torch_grad(
        outputs = loss,
        inputs = layer,
        grad_outputs = torch.ones_like(loss),
        retain_graph = True
    )[0].detach()

# ctvit - 3d ViT with factorized spatial and temporal attention made into an vqgan-vae autoencoder

def pick_video_frame(video, frame_indices):
    batch, device = video.shape[0], video.device
    video = rearrange(video, 'b c f ... -> b f c ...')
    device=torch.device('cuda')
    batch_indices = torch.arange(batch, device = device)
    batch_indices = rearrange(batch_indices, 'b -> b 1')
    images = video[batch_indices, frame_indices]
    images = rearrange(images, 'b 1 c ... -> b c ...')
    return images

class CTViT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        codebook_size,
        image_size,
        patch_size,
        temporal_patch_size,
        spatial_depth,
        temporal_depth,
        discr_base_dim = 16,
        dim_head = 64,
        heads = 8,
        channels = 1,
        use_vgg_and_gan = True,
        vgg = None,
        discr_attn_res_layers = (16,),
        use_hinge_loss = True,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        """
        einstein notations:

        b - batch
        c - channels
        t - time
        d - feature dimension
        p1, p2, pt - image patch sizes and then temporal patch size
        """

        super().__init__()
        
        # fix bug 
        self.use_vgg_and_gan = use_vgg_and_gan

        self.image_size = pair(image_size)
        self.patch_size = pair(patch_size)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        self.spatial_rel_pos_bias = ContinuousPositionBias(dim = dim, heads = heads)

        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0

        self.to_patch_emb_first_frame = nn.Sequential(
            Rearrange('b c 1 (h p1) (w p2) -> b 1 h w (c p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.LayerNorm(channels * patch_width * patch_height),
            nn.Linear(channels * patch_width * patch_height, dim),
            nn.LayerNorm(dim)
        )

        self.to_patch_emb = nn.Sequential(
            Rearrange('b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            nn.LayerNorm(dim)
        )

        transformer_kwargs = dict(
            dim = dim,
            dim_head = dim_head,
            heads = heads,
            attn_dropout = attn_dropout,
            ff_dropout = ff_dropout,
            peg = True,
            peg_causal = True,
        )
        self.enc_spatial_transformer = Transformer(depth = spatial_depth, **transformer_kwargs)
        self.enc_temporal_transformer = Transformer(depth = temporal_depth, **transformer_kwargs)
        self.vq = VectorQuantize(dim = dim, codebook_size = codebook_size, use_cosine_sim = True)

        self.to_pixels_first_frame = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height),
            Rearrange('b 1 h w (c p1 p2) -> b c 1 (h p1) (w p2)', p1 = patch_height, p2 = patch_width)
        )

        self.to_pixels = nn.Sequential(
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            Rearrange('b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)', p1 = patch_height, p2 = patch_width, pt = temporal_patch_size),
        )
        
        self.gen_loss = hinge_gen_loss if use_hinge_loss else bce_gen_loss

    def calculate_video_token_mask(self, videos, video_frame_mask):
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        first_frame_mask, rest_frame_mask = video_frame_mask[:, :1], video_frame_mask[:, 1:]
        rest_vq_mask = rearrange(rest_frame_mask, 'b (f p) -> b f p', p = self.temporal_patch_size)
        video_mask = torch.cat((first_frame_mask, rest_vq_mask.any(dim = -1)), dim = -1)
        return repeat(video_mask, 'b f -> b (f hw)', hw = (h // ph) * (w // pw))

    def get_video_patch_shape(self, num_frames, include_first_frame = True):
        patch_frames = 0

        if include_first_frame:
            num_frames -= 1
            patch_frames += 1

        patch_frames += (num_frames // self.temporal_patch_size)

        return (patch_frames, *self.patch_height_width)

    @property
    def image_num_tokens(self):
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def frames_per_num_tokens(self, num_tokens):
        tokens_per_frame = self.image_num_tokens

        assert (num_tokens % tokens_per_frame) == 0, f'number of tokens must be divisible by number of tokens per frame {tokens_per_frame}'
        assert (num_tokens > 0)

        pseudo_frames = num_tokens // tokens_per_frames
        return (pseudo_frames - 1) * self.temporal_patch_size + 1

    def num_tokens_per_frames(self, num_frames, include_first_frame = True):
        image_num_tokens = self.image_num_tokens

        total_tokens = 0

        if include_first_frame:
            num_frames -= 1
            total_tokens += image_num_tokens

        assert (num_frames % self.temporal_patch_size) == 0

        return total_tokens + int(num_frames / self.temporal_patch_size) * image_num_tokens

    def copy_for_eval(self):
        device = next(self.parameters()).device
        device=torch.device('cuda')
        vae_copy = copy.deepcopy(self.cpu())

        if vae_copy.use_vgg_and_gan:
            del vae_copy.discr
            del vae_copy.vgg

        vae_copy.eval()
        return vae_copy.to(device)

    #@remove_vgg
    def state_dict(self, *args, **kwargs):
        return super().state_dict(*args, **kwargs)

    #@remove_vgg
    def load_state_dict(self, *args, **kwargs):
        return super().load_state_dict(*args, **kwargs)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices):
        codes = self.vq.codebook[indices]
        return self.decode(codes)

    @property
    def patch_height_width(self):
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    def encode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        device=torch.device('cuda')
        
        attn_bias = self.spatial_rel_pos_bias(h, w, device = device)

        tokens = self.enc_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # encode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        tokens = self.enc_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        return tokens

    def decode(
        self,
        tokens
    ):
        b = tokens.shape[0]
        h, w = self.patch_height_width

        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        video_shape = tuple(tokens.shape[:-1])

        # decode - temporal

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        # tokens = self.dec_temporal_transformer(tokens, video_shape = video_shape)

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b = b, h = h, w = w)

        # decode - spatial

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
        device=torch.device('cuda')
        attn_bias = self.spatial_rel_pos_bias(h, w, device = device)

        # tokens = self.dec_spatial_transformer(tokens, attn_bias = attn_bias, video_shape = video_shape)

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b = b, h = h , w = w)

        # to pixels

        #first_frame_token, rest_frames_tokens = tokens[:, :1], tokens[:, 1:]

        #first_frame = self.to_pixels_first_frame(first_frame_token)

        #rest_frames = self.to_pixels(rest_frames_tokens)

        recon_video = self.to_pixels(tokens)

        #recon_video = torch.cat((first_frame, rest_frames), dim = 2)

        return recon_video

    def forward(
        self,
        video,
        mask = None,
        return_recons = False,
        return_recons_only = False,
        return_discr_loss = False,
        apply_grad_penalty = True,
        return_only_codebook_ids = False,
        return_encoded_tokens=True
    ):
        assert video.ndim in {4, 5}

        is_image = video.ndim == 4
        #print(video.shape)

        if is_image:
            video = rearrange(video, 'b c h w -> b c 1 h w')
            assert not exists(mask)

        b, c, f, *image_dims, device = *video.shape, video.device
        device=torch.device('cuda')
        
        assert tuple(image_dims) == self.image_size
        assert not exists(mask) or mask.shape[-1] == f

        first_frame, rest_frames = video[:, :, :1], video[:, :, 1:]

        # derive patches

        #first_frame_tokens = self.to_patch_emb_first_frame(first_frame)
        #rest_frames_tokens = self.to_patch_emb(rest_frames)
        #tokens = torch.cat((first_frame_tokens, rest_frames_tokens), dim = 1)
        tokens = self.to_patch_emb(video)
        # save height and width in

        shape = tokens.shape
        *_, h, w, _ = shape

        # encode - spatial

        tokens = self.encode(tokens)

        # quantize

        tokens, packed_fhw_shape = pack([tokens], 'b * d')

        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)
            print(vq_mask.dtype)

        
        tokens, indices, commit_loss = self.vq(tokens, mask = vq_mask)

        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h = h, w = w)

        if return_encoded_tokens:
            # tokens = torch.mean(tokens, dim=1)
            # tokens = tokens.view(tokens.shape[0], -1)
            tokens = tokens.view(tokens.shape[0], tokens.shape[1], -1)
            return tokens
            
        recon_video = self.decode(tokens)

        returned_recon = rearrange(recon_video, 'b c 1 h w -> b c h w') if is_image else recon_video.clone()

        if return_recons_only:
            return returned_recon

        if exists(mask):
            # variable lengthed video / images training
            recon_loss = F.mse_loss(video, recon_video, reduction = 'none')
            recon_loss = recon_loss[repeat(mask, 'b t -> b c t', c = c)]
            recon_loss = recon_loss.mean()
        else:
            recon_loss = F.mse_loss(video, recon_video)

        # prepare a random frame index to be chosen for discriminator and perceptual loss

        pick_frame_logits = torch.randn(b, f)

        if exists(mask):
            mask_value = -torch.finfo(pick_frame_logits.dtype).max
            pick_frame_logits = pick_frame_logits.masked_fill(~mask, mask_value)

        frame_indices = pick_frame_logits.topk(1, dim = -1).indices

        # whether to return discriminator loss

        if return_discr_loss:
            assert exists(self.discr), 'discriminator must exist to train it'

            video = pick_video_frame(video, frame_indices)
            recon_video = pick_video_frame(recon_video, frame_indices)

            recon_video = recon_video.detach()
            video.requires_grad_()

            transform = T.Compose([T.Resize(256)])

            recon_video = transform(recon_video)
            video = transform(video)

            
            #print("TEST")
            #print(recon_video.shape)


            recon_video_discr_logits, video_discr_logits = map(self.discr, (recon_video, video))

            discr_loss = self.discr_loss(recon_video_discr_logits, video_discr_logits)

            if apply_grad_penalty:
                gp = gradient_penalty(video, video_discr_logits)
                loss = discr_loss + gp

            if return_recons:
                return loss, returned_recon

            return loss

        # early return if training on grayscale

        if not self.use_vgg_and_gan:
            if return_recons:
                return recon_loss, returned_recon

            return recon_loss

        # perceptual loss

        input_vgg_input = pick_video_frame(video, frame_indices)
        recon_vgg_input = pick_video_frame(recon_video, frame_indices)
        transform = T.Compose([T.Resize(256)])
        input_vgg_input = transform(input_vgg_input)
        recon_vgg_input=transform(recon_vgg_input)
        
        # handle grayscale for vgg
        
        if video.shape[1] == 1:
            input_vgg_input2, recon_vgg_input2 = map(lambda t: repeat(t, 'b 1 ... -> b c ...', c = 3), (input_vgg_input, recon_vgg_input))
        transform = T.Compose([T.Resize(256)])
        input_vgg_input2 = transform(input_vgg_input2)
        recon_vgg_input2 = transform(recon_vgg_input2)


        
        input_vgg_feats = self.vgg(input_vgg_input2)
        recon_vgg_feats = self.vgg(recon_vgg_input2)

        perceptual_loss = F.mse_loss(input_vgg_feats, recon_vgg_feats)

        # generator loss

        gen_loss = self.gen_loss(self.discr(recon_vgg_input))

        # calculate adaptive weight

        last_dec_layer = self.to_pixels[0].weight

        norm_grad_wrt_gen_loss = grad_layer_wrt_loss(gen_loss, last_dec_layer).norm(p = 2)
        norm_grad_wrt_perceptual_loss = grad_layer_wrt_loss(perceptual_loss, last_dec_layer).norm(p = 2)

        adaptive_weight = safe_div(norm_grad_wrt_perceptual_loss, norm_grad_wrt_gen_loss)
        adaptive_weight.clamp_(max = 1e4)

        # combine losses

        loss = recon_loss + perceptual_loss + commit_loss + adaptive_weight * gen_loss

        if return_recons:
            return loss, returned_recon

        return loss


# --------------------------------

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum

from beartype import beartype
from typing import Tuple

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def leaky_relu(p = 0.1):
    return nn.LeakyReLU(p)

def l2norm(t):
    return F.normalize(t, dim = -1)

# bias-less layernorm, being used in more recent T5s, PaLM, also in @borisdayma 's experiments shared with me
# greater stability

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# feedforward

class GEGLU(nn.Module):
    def forward(self, x):
        x, gate = x.chunk(2, dim = -1)
        return F.gelu(gate) * x

def FeedForward(dim, mult = 4, dropout = 0.):
    inner_dim = int(mult * (2 / 3) * dim)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim * 2, bias = False),
        GEGLU(),
        nn.Dropout(dropout),
        nn.Linear(inner_dim, dim, bias = False)
    )

# PEG - position generating module

class PEG(nn.Module):
    def __init__(self, dim, causal = False):
        super().__init__()
        self.causal = causal
        self.dsconv = nn.Conv3d(dim, dim, 3, groups = dim)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        if needs_shape:
            x = x.reshape(*shape, -1)

        x = rearrange(x, 'b ... d -> b d ...')

        frame_padding = (2, 0) if self.causal else (1, 1)

        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value = 0.)
        x = self.dsconv(x)

        x = rearrange(x, 'b d ... -> b ... d')

        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)

# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        dim_context = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        num_null_kv = 0,
        norm_context = True,
        dropout = 0.,
        scale = 8
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads = heads)

        self.attn_dropout = nn.Dropout(dropout)

        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias = False)

        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(
        self,
        x,
        mask = None,
        context = None,
        attn_bias = None
    ):
        batch, device, dtype = x.shape[0], x.device, x.dtype
        device=torch.device('cuda')
        if exists(context):
            context = self.context_norm(context)

        kv_input = default(context, x)

        x = self.norm(x)

        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b = batch, r = 2).unbind(dim = -2)

        k = torch.cat((nk, k), dim = -2)
        v = torch.cat((nv, v), dim = -2)

        q, k = map(l2norm, (q, k))
        q = q * self.q_scale
        k = k * self.k_scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]

        if exists(attn_bias):
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value = 0.)
            sim = sim + attn_bias

        if exists(mask):
            mask = F.pad(mask, (self.num_null_kv, 0), value = True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        if self.causal:
            sim = sim + self.rel_pos_bias(sim)
            device=torch.device('cuda')
            causal_mask = torch.ones((i, j), device = device, dtype = torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)
        attn = self.attn_dropout(attn)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# alibi positional bias for extrapolation

class AlibiPositionalBias(nn.Module):
    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        self.register_buffer('slopes', slopes, persistent = False)
        self.register_buffer('bias', None, persistent = False)

    def get_bias(self, i, j, device):
        device=torch.device('cuda')
        i_arange = torch.arange(j - i, j, device = device)
        j_arange = torch.arange(j, device = device)
        bias = -torch.abs(rearrange(j_arange, 'j -> 1 1 j') - rearrange(i_arange, 'i -> 1 i 1'))
        return bias

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2]

    def forward(self, sim):
        h, i, j, device = *sim.shape[-3:], sim.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]
        device=torch.device('cuda')
        bias = self.get_bias(i, j, device)
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))
        self.register_buffer('bias', bias, persistent = False)

        return self.bias

class ContinuousPositionBias(nn.Module):
    """ from https://arxiv.org/abs/2111.09883 """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims = 2, # 2 for images, 3 for video
        layers = 2,
        log_dist = True,
        cache_rel_pos = False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        self.net = nn.ModuleList([])
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent = False)

    def forward(self, *dimensions, device = torch.device('cpu')):

        if not exists(self.rel_pos) or not self.cache_rel_pos:
            device=torch.device('cuda')
            positions = [torch.arange(d, device = device) for d in dimensions]
            grid = torch.stack(torch.meshgrid(*positions, indexing = 'ij'))
            grid = rearrange(grid, 'c ... -> (...) c')
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            self.register_buffer('rel_pos', rel_pos, persistent = False)

        rel_pos = self.rel_pos.to(torch.bfloat16)

        for layer in self.net:
            # rel_pos = layer(rel_pos.float())
            rel_pos = layer(rel_pos)

        return rearrange(rel_pos, 'i j h -> h i j')

# transformer

class Transformer(nn.Module):
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context = None,
        causal = False,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        peg = False,
        peg_causal = False,
        attn_num_null_kv = 2,
        has_cross_attn = False,
        attn_dropout = 0.,
        ff_dropout = 0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PEG(dim = dim, causal = peg_causal) if peg else None,
                Attention(dim = dim, dim_head = dim_head, heads = heads, causal = causal, dropout = attn_dropout),
                Attention(dim = dim, dim_head = dim_head, dim_context = dim_context, heads = heads, causal = False, num_null_kv = attn_num_null_kv, dropout = attn_dropout) if has_cross_attn else None,
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
            ]))

        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias = None,
        context = None,
        self_attn_mask = None,
        cross_attn_context_mask = None
    ):

        for peg, self_attn, cross_attn, ff in self.layers:
            if exists(peg):
                x = peg(x, shape = video_shape) + x

            x = self_attn(x, attn_bias = attn_bias, mask = self_attn_mask) + x

            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context = context, mask = cross_attn_context_mask) + x

            x = ff(x) + x

        return self.norm_out(x)




import torch
from torch import nn, einsum
import torch.nn.functional as F
import torch.distributed as distributed
from torch.cuda.amp import autocast

from einops import rearrange, repeat, pack, unpack
from contextlib import contextmanager

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def noop(*args, **kwargs):
    pass

def l2norm(t):
    return F.normalize(t, p = 2, dim = -1)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def uniform_init(*shape):
    t = torch.empty(shape)
    nn.init.kaiming_uniform_(t)
    return t

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    if temperature == 0:
        return t.argmax(dim = dim)

    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def laplace_smoothing(x, n_categories, eps = 1e-5):
    return (x + eps) / (x.sum() + n_categories * eps)

def sample_vectors(samples, num):
    num_samples, device = samples.shape[0], samples.device
    if num_samples >= num:
        indices = torch.randperm(num_samples, device = device)[:num]
    else:
        indices = torch.randint(0, num_samples, (num,), device = device)

    return samples[indices]

def batched_sample_vectors(samples, num):
    return torch.stack([sample_vectors(sample, num) for sample in samples.unbind(dim = 0)], dim = 0)

def pad_shape(shape, size, dim = 0):
    return [size if i == dim else s for i, s in enumerate(shape)]

def sample_multinomial(total_count, probs):
    device = probs.device
    probs = probs.cpu()

    total_count = probs.new_full((), total_count)
    remainder = probs.new_ones(())
    sample = torch.empty_like(probs, dtype = torch.long)

    for i, p in enumerate(probs):
        s = torch.binomial(total_count, p / remainder)
        sample[i] = s
        total_count -= s
        remainder -= p

    return sample.to(device)

def all_gather_sizes(x, dim):
    size = torch.tensor(x.shape[dim], dtype = torch.long, device = x.device)
    all_sizes = [torch.empty_like(size) for _ in range(distributed.get_world_size())]
    distributed.all_gather(all_sizes, size)
    return torch.stack(all_sizes)

def all_gather_variably_sized(x, sizes, dim = 0):
    rank = distributed.get_rank()
    all_x = []

    for i, size in enumerate(sizes):
        t = x if i == rank else x.new_empty(pad_shape(x.shape, size, dim))
        distributed.broadcast(t, src = i, async_op = True)
        all_x.append(t)

    distributed.barrier()
    return all_x

def sample_vectors_distributed(local_samples, num):
    local_samples = rearrange(local_samples, '1 ... -> ...')

    rank = distributed.get_rank()
    all_num_samples = all_gather_sizes(local_samples, dim = 0)

    if rank == 0:
        samples_per_rank = sample_multinomial(num, all_num_samples / all_num_samples.sum())
    else:
        samples_per_rank = torch.empty_like(all_num_samples)

    distributed.broadcast(samples_per_rank, src = 0)
    samples_per_rank = samples_per_rank.tolist()

    local_samples = sample_vectors(local_samples, samples_per_rank[rank])
    all_samples = all_gather_variably_sized(local_samples, samples_per_rank, dim = 0)
    out = torch.cat(all_samples, dim = 0)

    return rearrange(out, '... -> 1 ...')

def batched_bincount(x, *, minlength):
    batch, dtype, device = x.shape[0], x.dtype, x.device
    target = torch.zeros(batch, minlength, dtype = dtype, device = device)
    values = torch.ones_like(x)
    target.scatter_add_(-1, x, values)
    return target

def kmeans(
    samples,
    num_clusters,
    num_iters = 10,
    use_cosine_sim = False,
    sample_fn = batched_sample_vectors,
    all_reduce_fn = noop
):
    num_codebooks, dim, dtype, device = samples.shape[0], samples.shape[-1], samples.dtype, samples.device

    means = sample_fn(samples, num_clusters)

    for _ in range(num_iters):
        if use_cosine_sim:
            dists = samples @ rearrange(means, 'h n d -> h d n')
        else:
            dists = -torch.cdist(samples, means, p = 2)

        buckets = torch.argmax(dists, dim = -1)
        bins = batched_bincount(buckets, minlength = num_clusters)
        all_reduce_fn(bins)

        zero_mask = bins == 0
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        new_means = buckets.new_zeros(num_codebooks, num_clusters, dim, dtype = dtype)

        new_means.scatter_add_(1, repeat(buckets, 'h n -> h n d', d = dim), samples)
        new_means = new_means / rearrange(bins_min_clamped, '... -> ... 1')
        all_reduce_fn(new_means)

        if use_cosine_sim:
            new_means = l2norm(new_means)

        means = torch.where(
            rearrange(zero_mask, '... -> ... 1'),
            means,
            new_means
        )

    return means, bins

def batched_embedding(indices, embeds):
    batch, dim = indices.shape[1], embeds.shape[-1]
    indices = repeat(indices, 'h b n -> h b n d', d = dim)
    embeds = repeat(embeds, 'h c d -> h b c d', b = batch)
    return embeds.gather(2, indices)

# regularization losses

def orthogonal_loss_fn(t):
    # eq (2) from https://arxiv.org/abs/2112.00384
    h, n = t.shape[:2]
    normed_codes = l2norm(t)
    cosine_sim = einsum('h i d, h j d -> h i j', normed_codes, normed_codes)
    return (cosine_sim ** 2).sum() / (h * n ** 2) - (1 / n)

# distance types

class EuclideanCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        use_ddp = False,
        learnable_codebook = False,
        sample_codebook_temp = 0
    ):
        super().__init__()
        self.decay = decay
        init_fn = uniform_init if not kmeans_init else torch.zeros
        embed = init_fn(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        assert not (use_ddp and num_codebooks > 1 and kmeans_init), 'kmeans init is not compatible with multiple codebooks in distributed environment for now'

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))
        self.register_buffer('embed_avg', embed.clone())

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.embed_avg.data.copy_(embed.clone())
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype
        flatten = rearrange(x, 'h ... d -> h (...) d')

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()

        dist = -torch.cdist(flatten, embed, p = 2)

        embed_ind = gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            cluster_size = embed_onehot.sum(dim = 1)

            self.all_reduce_fn(cluster_size)
            self.cluster_size.data.lerp_(cluster_size, 1 - self.decay)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum.contiguous())
            self.embed_avg.data.lerp_(embed_sum, 1 - self.decay)

            cluster_size = laplace_smoothing(self.cluster_size, self.codebook_size, self.eps) * self.cluster_size.sum()

            embed_normalized = self.embed_avg / rearrange(cluster_size, '... -> ... 1')
            self.embed.data.copy_(embed_normalized)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind

class CosineSimCodebook(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        num_codebooks = 1,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        decay = 0.8,
        eps = 1e-5,
        threshold_ema_dead_code = 2,
        use_ddp = False,
        learnable_codebook = False,
        sample_codebook_temp = 0.
    ):
        super().__init__()
        self.decay = decay

        if not kmeans_init:
            embed = l2norm(uniform_init(num_codebooks, codebook_size, dim))
        else:
            embed = torch.zeros(num_codebooks, codebook_size, dim)

        self.codebook_size = codebook_size
        self.num_codebooks = num_codebooks

        self.kmeans_iters = kmeans_iters
        self.eps = eps
        self.threshold_ema_dead_code = threshold_ema_dead_code
        self.sample_codebook_temp = sample_codebook_temp

        self.sample_fn = sample_vectors_distributed if use_ddp and sync_kmeans else batched_sample_vectors
        self.kmeans_all_reduce_fn = distributed.all_reduce if use_ddp and sync_kmeans else noop
        self.all_reduce_fn = distributed.all_reduce if use_ddp else noop

        self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        self.register_buffer('cluster_size', torch.zeros(num_codebooks, codebook_size))

        self.learnable_codebook = learnable_codebook
        if learnable_codebook:
            self.embed = nn.Parameter(embed)
        else:
            self.register_buffer('embed', embed)

    @torch.jit.ignore
    def init_embed_(self, data):
        if self.initted:
            return

        embed, cluster_size = kmeans(
            data,
            self.codebook_size,
            self.kmeans_iters,
            use_cosine_sim = True,
            sample_fn = self.sample_fn,
            all_reduce_fn = self.kmeans_all_reduce_fn
        )

        self.embed.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)
        self.initted.data.copy_(torch.Tensor([True]))

    def replace(self, batch_samples, batch_mask):
        batch_samples = l2norm(batch_samples)

        for ind, (samples, mask) in enumerate(zip(batch_samples.unbind(dim = 0), batch_mask.unbind(dim = 0))):
            if not torch.any(mask):
                continue

            sampled = self.sample_fn(rearrange(samples, '... -> 1 ...'), mask.sum().item())
            self.embed.data[ind][mask] = rearrange(sampled, '1 ... -> ...')

    def expire_codes_(self, batch_samples):
        if self.threshold_ema_dead_code == 0:
            return

        expired_codes = self.cluster_size < self.threshold_ema_dead_code

        if not torch.any(expired_codes):
            return

        batch_samples = rearrange(batch_samples, 'h ... d -> h (...) d')
        self.replace(batch_samples, batch_mask = expired_codes)

    @autocast(enabled = False)
    def forward(self, x):
        needs_codebook_dim = x.ndim < 4

        # x = x.float()

        if needs_codebook_dim:
            x = rearrange(x, '... -> 1 ...')

        shape, dtype = x.shape, x.dtype

        flatten = rearrange(x, 'h ... d -> h (...) d')
        flatten = l2norm(flatten)

        self.init_embed_(flatten)

        embed = self.embed if not self.learnable_codebook else self.embed.detach()
        embed = l2norm(embed)

        dist = einsum('h n d, h c d -> h n c', flatten, embed)
        embed_ind = gumbel_sample(dist, dim = -1, temperature = self.sample_codebook_temp)
        embed_onehot = F.one_hot(embed_ind, self.codebook_size).type(dtype)
        embed_ind = embed_ind.view(*shape[:-1])

        quantize = batched_embedding(embed_ind, self.embed)

        if self.training:
            bins = embed_onehot.sum(dim = 1)
            self.all_reduce_fn(bins)

            self.cluster_size.data.lerp_(bins, 1 - self.decay)

            zero_mask = (bins == 0)
            bins = bins.masked_fill(zero_mask, 1.)

            embed_sum = einsum('h n d, h n c -> h c d', flatten, embed_onehot)
            self.all_reduce_fn(embed_sum)

            embed_normalized = embed_sum / rearrange(bins, '... -> ... 1')
            embed_normalized = l2norm(embed_normalized)

            embed_normalized = torch.where(
                rearrange(zero_mask, '... -> ... 1'),
                embed,
                embed_normalized
            )

            self.embed.data.lerp_(embed_normalized, 1 - self.decay)
            self.expire_codes_(x)

        if needs_codebook_dim:
            quantize, embed_ind = map(lambda t: rearrange(t, '1 ... -> ...'), (quantize, embed_ind))

        return quantize, embed_ind

# main class

class VectorQuantize(nn.Module):
    def __init__(
        self,
        dim,
        codebook_size,
        codebook_dim = None,
        heads = 1,
        separate_codebook_per_head = False,
        decay = 0.8,
        eps = 1e-5,
        kmeans_init = False,
        kmeans_iters = 10,
        sync_kmeans = True,
        use_cosine_sim = False,
        threshold_ema_dead_code = 0,
        channel_last = True,
        accept_image_fmap = False,
        commitment_weight = 1.,
        orthogonal_reg_weight = 0.,
        orthogonal_reg_active_codes_only = False,
        orthogonal_reg_max_codes = None,
        sample_codebook_temp = 0.,
        sync_codebook = False
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.separate_codebook_per_head = separate_codebook_per_head

        codebook_dim = default(codebook_dim, dim)
        codebook_input_dim = codebook_dim * heads

        requires_projection = codebook_input_dim != dim
        self.project_in = nn.Linear(dim, codebook_input_dim) if requires_projection else nn.Identity()
        self.project_out = nn.Linear(codebook_input_dim, dim) if requires_projection else nn.Identity()

        self.eps = eps
        self.commitment_weight = commitment_weight

        has_codebook_orthogonal_loss = orthogonal_reg_weight > 0
        self.orthogonal_reg_weight = orthogonal_reg_weight
        self.orthogonal_reg_active_codes_only = orthogonal_reg_active_codes_only
        self.orthogonal_reg_max_codes = orthogonal_reg_max_codes

        codebook_class = EuclideanCodebook if not use_cosine_sim else CosineSimCodebook

        self._codebook = codebook_class(
            dim = codebook_dim,
            num_codebooks = heads if separate_codebook_per_head else 1,
            codebook_size = codebook_size,
            kmeans_init = kmeans_init,
            kmeans_iters = kmeans_iters,
            sync_kmeans = sync_kmeans,
            decay = decay,
            eps = eps,
            threshold_ema_dead_code = threshold_ema_dead_code,
            use_ddp = sync_codebook,
            learnable_codebook = has_codebook_orthogonal_loss,
            sample_codebook_temp = sample_codebook_temp
        )

        self.codebook_size = codebook_size

        self.accept_image_fmap = accept_image_fmap
        self.channel_last = channel_last

    @property
    def codebook(self):
        codebook = self._codebook.embed
        if self.separate_codebook_per_head:
            return codebook

        return rearrange(codebook, '1 ... -> ...')

    def get_codes_from_indices(self, indices):
        codebook = self.codebook
        is_multiheaded = codebook.ndim > 2

        if not is_multiheaded:
            codes = codebook[indices]
            return rearrange(codes, '... h d -> ... (h d)')

        indices, ps = pack([indices], 'b * h')
        indices = rearrange(indices, 'b n h -> b h n')

        indices = repeat(indices, 'b h n -> b h n d', d = codebook.shape[-1])
        codebook = repeat(codebook, 'h n d -> b h n d', b = indices.shape[0])

        codes = codebook.gather(2, indices)
        codes = rearrange(codes, 'b h n d -> b n (h d)')
        codes, = unpack(codes, ps, 'b * d')
        return codes

    def forward(
        self,
        x,
        mask = None
    ):
        only_one = x.ndim == 2

        if only_one:
            x = rearrange(x, 'b d -> b 1 d')

        shape, device, heads, is_multiheaded, codebook_size = x.shape, x.device, self.heads, self.heads > 1, self.codebook_size

        need_transpose = not self.channel_last and not self.accept_image_fmap

        if self.accept_image_fmap:
            height, width = x.shape[-2:]
            x = rearrange(x, 'b c h w -> b (h w) c')

        if need_transpose:
            x = rearrange(x, 'b d n -> b n d')

        x = self.project_in(x)

        if is_multiheaded:
            ein_rhs_eq = 'h b n d' if self.separate_codebook_per_head else '1 (b h) n d'
            x = rearrange(x, f'b n (h d) -> {ein_rhs_eq}', h = heads)
        # x = x.to(torch.float32)
        quantize, embed_ind = self._codebook(x)
        # quantize = quantize.to(torch.bfloat16)
        if self.training:
            quantize = x + (quantize - x).detach()

        loss = torch.tensor([0.], device = device, requires_grad = self.training)

        if self.training:
            if self.commitment_weight > 0:
                detached_quantize = quantize.detach()

                if exists(mask):
                    # with variable lengthed sequences
                    commit_loss = F.mse_loss(detached_quantize, x, reduction = 'none')

                    if is_multiheaded:
                        mask = repeat(mask, 'b n -> c (b h) n', c = commit_loss.shape[0], h = commit_loss.shape[1] // mask.shape[0])

                    commit_loss = commit_loss[mask].mean()
                else:
                    commit_loss = F.mse_loss(detached_quantize, x)

                loss = loss + commit_loss * self.commitment_weight

            if self.orthogonal_reg_weight > 0:
                codebook = self._codebook.embed

                if self.orthogonal_reg_active_codes_only:
                    # only calculate orthogonal loss for the activated codes for this batch
                    unique_code_ids = torch.unique(embed_ind)
                    codebook = codebook[unique_code_ids]

                num_codes = codebook.shape[0]
                if exists(self.orthogonal_reg_max_codes) and num_codes > self.orthogonal_reg_max_codes:
                    rand_ids = torch.randperm(num_codes, device = device)[:self.orthogonal_reg_max_codes]
                    codebook = codebook[rand_ids]

                orthogonal_reg_loss = orthogonal_loss_fn(codebook)
                loss = loss + orthogonal_reg_loss * self.orthogonal_reg_weight

        if is_multiheaded:
            if self.separate_codebook_per_head:
                quantize = rearrange(quantize, 'h b n d -> b n (h d)', h = heads)
                embed_ind = rearrange(embed_ind, 'h b n -> b n h', h = heads)
            else:
                quantize = rearrange(quantize, '1 (b h) n d -> b n (h d)', h = heads)
                embed_ind = rearrange(embed_ind, '1 (b h) n -> b n h', h = heads)

        quantize = self.project_out(quantize)

        if need_transpose:
            quantize = rearrange(quantize, 'b n d -> b d n')

        if self.accept_image_fmap:
            quantize = rearrange(quantize, 'b (h w) c -> b c h w', h = height, w = width)
            embed_ind = rearrange(embed_ind, 'b (h w) ... -> b h w ...', h = height, w = width)

        if only_one:
            quantize = rearrange(quantize, 'b 1 d -> b d')
            embed_ind = rearrange(embed_ind, 'b 1 -> b')

        return quantize, embed_ind, loss