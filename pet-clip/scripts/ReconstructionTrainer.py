from pathlib import Path
from shutil import rmtree
from datetime import timedelta

from transformer_maskgit.optimizer import get_optimizer
from transformers import BertTokenizer, BertModel



import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler

from data_new import *

from einops import rearrange
import accelerate
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs

import math
import torch.optim.lr_scheduler as lr_scheduler
# from ct_clip import CTCLIP
from ctvit import CTViT


# helpers
def apply_softmax(array):
    softmax = torch.nn.Softmax(dim=0)
    softmax_array = softmax(array)
    return softmax_array

def exists(val):
    return val is not None

def noop(*args, **kwargs):
    pass

def cycle(dl):
    while True:
        for data in dl:
            yield data

def yes_or_no(question):
    answer = input(f'{question} (y/n) ')
    return answer.lower() in ('yes', 'y')

# def accum_log(log, new_logs):
#     for key, new_value in new_logs.items():
#         old_value = log.get(key, 0.)
#         log[key] = old_value + new_value
#     return log

class CosineAnnealingWarmUpRestarts(lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_warmup=10000, gamma=1.0, last_epoch=-1):
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_max = eta_max
        self.T_warmup = T_warmup
        self.gamma = gamma
        self.T_cur = 0
        self.lr_min = 0
        self.iteration = 0

        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.iteration < self.T_warmup:
            lr = self.eta_max * self.iteration / self.T_warmup
        else:
            self.T_cur = self.iteration - self.T_warmup
            T_i = self.T_0
            while self.T_cur >= T_i:
                self.T_cur -= T_i
                T_i *= self.T_mult
                self.lr_min = self.eta_max * (self.gamma ** self.T_cur)
            lr = self.lr_min + 0.5 * (self.eta_max - self.lr_min) * \
                 (1 + math.cos(math.pi * self.T_cur / T_i))

        self.iteration += 1
        return [lr for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        self._update_lr()
        self._update_T()

    def _update_lr(self):
        self.optimizer.param_groups[0]['lr'] = self.get_lr()[0]

    def _update_T(self):
        if self.T_cur == self.T_0:
            self.T_cur = 0
            self.lr_min = 0
            self.iteration = 0
            self.T_0 *= self.T_mult
            self.eta_max *= self.gamma

class ReconstructionTrainer(nn.Module):
    def __init__(
        self,
        CTViT: CTViT,
        *,
        num_train_steps,
        batch_size,
        root = './',
        lr = 1.25e-6,
        logger,
        wd = 0.,
        max_grad_norm = 0.5,
        save_results_every = 1000,
        save_model_every = 1000 ,
        results_folder = './',
        num_workers = 8,
        accelerate_kwargs: dict = dict()
    ):
        super().__init__()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs, kwargs], **accelerate_kwargs)
        self.CTViT = CTViT

        self.register_buffer('steps', torch.Tensor([0]))

        self.num_train_steps = num_train_steps
        self.batch_size = batch_size

        all_parameters = set(CTViT.parameters())

        self.optim = get_optimizer(all_parameters, lr=lr, wd=wd)

        self.max_grad_norm = max_grad_norm
        self.lr=lr
        # Load the pre-trained weights
    
        self.ds = MedicalImageReportDataset(root=root, paraphase=True,
                                             augmentation=None, split='train')

        self.valid_ds = MedicalImageReportDataset(root=root,paraphase=False, augmentation=None, split='val')


        self.dl = DataLoader(
            self.ds,
            num_workers=num_workers,
            batch_size=self.batch_size,
            shuffle = True,
        )

        self.valid_dl = DataLoader(
            self.valid_ds,
            num_workers=num_workers,
            batch_size=1,
            shuffle = False,
        )

        # prepare with accelerator
        self.dl_iter=cycle(self.dl)
        self.valid_dl_iter=cycle(self.valid_dl)
        self.device = self.accelerator.device
        self.CTViT.to(self.device)

        (
 			self.dl_iter,
            self.valid_dl_iter,
            self.CTViT,
            self.optim,
        ) = self.accelerator.prepare(
            self.dl_iter,
            self.valid_dl_iter,
            self.CTViT,
            self.optim,
        )

        self.save_model_every = save_model_every
        self.save_results_every = save_results_every

        self.results_folder = Path(results_folder)

        # if len([*self.results_folder.glob('**/*')]) > 0 and yes_or_no('do you want to clear previous experiment checkpoints and results?'):
        #     rmtree(str(self.results_folder))

        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.logger = logger



    def save(self, path):
        if not self.accelerator.is_local_main_process:
            return

        pkg = dict(
            model=self.accelerator.get_state_dict(self.CTViT),
            optim=self.optim.state_dict(),
        )
        torch.save(pkg, path)

    def load(self, path):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(path)

        CTViT = self.accelerator.unwrap_model(self.CTViT)
        CTViT.load_state_dict(pkg['model'])

        self.optim.load_state_dict(pkg['optim'])

    def print(self, msg):
        self.accelerator.print(msg)


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def train_step(self):
        device = self.device

        steps = int(self.steps.item())

        self.CTViT.train()

        # update CTClip model
        video = next(self.dl_iter)
        device=self.device
        video=video.to(device)
        mask = torch.ones((video.shape[0], video.shape[2])).bool().to(device)
        #text = text.to(device)
        # text = list(text)
        # text_tokens=self.tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=258).to(device)

        #video = video
        with self.accelerator.autocast():
            loss, reconstruction = self.CTViT(video)
        
        

        self.accelerator.backward(loss)
        if exists(self.max_grad_norm):
            self.accelerator.clip_grad_norm_(self.CTViT.parameters(), self.max_grad_norm)

        self.optim.step()
        self.optim.zero_grad()
        # self.print(f"{steps}: loss: {logs['loss']}")
        #  write logs to txt file 
        
        # with open(self.results_folder / '0_logs.txt', 'a') as f:
        #     f.write(f"{steps}: loss: {logs['loss']}\n")
        self.logger.log('loss', loss.item())
        self.logger.log('steps', steps)
        # log lr 
        self.logger.log('lr', self.optim.param_groups[0]['lr'])
        self.logger.log_to_wandb()
        

        if self.is_main and not (steps % self.save_model_every): 
            gt_img = video[0, 60, :, :, :]
            recon_img = reconstruction[0, 60, :, :, :]
            self.logger.log_img('gt_img_train', gt_img, section='train')
            self.logger.log_img('recon_img_train', recon_img, section='train')
        
        if self.is_main and not (steps % self.save_results_every):
            with torch.no_grad():
                self.CTViT.eval()
                video_valid = next(self.valid_dl_iter)
                device=self.device
                video_valid=video_valid.to(device)
                with self.accelerator.autocast():
                    loss, reconstruction = self.CTViT(video_valid)
                self.logger.log_img('gt_img_valid', video_valid[0, 60, :, :, :], section='valid')
                self.logger.log_img('recon_img_valid', reconstruction[0, 60, :, :, :], section='valid')
                self.logger.log_validation_loss(loss.item(), section='valid')
                self.CTViT.train()

        if self.is_main and not (steps % self.save_model_every):
            model_path = str(self.results_folder / f'CTViT.{steps}.pt')
            state_dict=self.accelerator.get_state_dict(self.CTViT, unwrap=False)

            self.accelerator.save(state_dict, model_path)

            self.print(f'{steps}: saving model to {str(self.results_folder)}')


        self.steps += 1

    def train(self):
        device = next(self.CTViT.parameters()).device
        device=torch.device('cuda')
        while self.steps < self.num_train_steps:
            self.train_step()

        self.print('training complete')
