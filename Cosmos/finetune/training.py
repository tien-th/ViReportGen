import sys
sys.path.append("../")
from cosmos1.models.tokenizer.networks import TokenizerConfigs, TokenizerModels
import torch
from tqdm import tqdm
from dataset_loader import get_dataloader
from loguru import logger
from loss import SSIM3D
import numpy as np
import matplotlib.pyplot as plt
import os 
import yaml
model_class = TokenizerModels.CV
config = TokenizerConfigs.CV8x8x8.value


def setup_model():
    pass


def train_1_epoch(model, 
                  train_loader, 
                  val_loader, 
                  device,
                  optimizer, 
                  batch,
                  logger,
                  loss_L1,
                  loss_SSIM = None
                  ):
    model.train()
    for batch in tqdm(train_loader):
        batch = batch.to(device)
        batch = batch.permute(0, 4, 1, 2, 3)  
        output = model(batch)[0]
        output = output[:, :, :120, :, :]

        l1 = loss_L1(output, batch)
        if loss_SSIM is not None:
            ssim = loss_SSIM(output, batch)
            loss = l1 + ssim 
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logger.info(f'train_loss: {loss.item()}')
        logger.info(f'train_l1: {l1.item()}')
        if loss_SSIM is not None:
            logger.info(f'train_ssim: {ssim.item()}')
    
    model.eval()
    with torch.no_grad():
        i = 0
        for batch in tqdm(val_loader):
            i += 1
            batch = batch.to(device)
            batch = batch.permute(0, 4, 1, 2, 3)  
            output = model(batch)[0]
            output = output[:, :, :120, :, :]
            l1 = loss_L1(output, batch)
            if loss_SSIM is not None:
                ssim = loss_SSIM(output, batch)
                loss = l1 + ssim
            logger.info(f'val_loss: {loss.item()}')
            logger.info(f'val_l1: {l1.item()}')
            if loss_SSIM is not None:
                logger.info(f'val_ssim: {ssim.item()}')

def training(cfg):
    logger.add(os.path.join("log", cfg["pretrained_path"].split("/")[-2], "training_logs_new.log"), rotation="100 MB", level="INFO", backtrace=True, diagnose=True)
    model = torch.jit.load(cfg['pretrained_path'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = get_dataloader(cfg["train"], mode = "train")
    val_loader = get_dataloader(cfg["val"], mode = "val")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    loss_L1 = torch.nn.L1Loss()
    if cfg['ssim']:
        loss_SSIM = SSIM3D()
    
    for epoch in range(cfg["EPOCH"]):
        logger.info(f"Epoch {epoch + 1}/{cfg['EPOCH']} - Training Started")
        train_1_epoch(model, train_loader, val_loader, device, optimizer, epoch, logger, loss_L1, loss_SSIM)
        logger.info(f"Epoch {epoch + 1}/{cfg['EPOCH']} - Training Finished")
        
        # Save model as JIT
        model.eval()
        scripted_model = torch.jit.script(model)  # or torch.jit.trace(model, example_input) if needed
        path_save = os.path.join(cfg["ckpt"], cfg["pretrained_path"].split("/")[-2])
        if not os.path.exists(path_save):
            os.makedirs(path_save)
        model_save_path = os.path.join(cfg['ckpt'], cfg["pretrained_path"].split("/")[-2], f"model_epoch_{epoch+1}.jit")
        scripted_model.save(model_save_path)
        logger.info(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    training(cfg)
    