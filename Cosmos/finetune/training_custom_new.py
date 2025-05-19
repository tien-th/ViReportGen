import sys
sys.path.append("../")
from cosmos1.models.tokenizer.networks import TokenizerConfigs, TokenizerModels
import torch
from tqdm import tqdm
from dataset_loader import get_dataloader
from loguru import logger
from loss import SSIM3D
import os
import yaml
import wandb
import random
import numpy as np

model_class = TokenizerModels.CV.value
config = TokenizerConfigs.CV8x8x8.value

def train_1_epoch(model, train_loader, val_loader, device, optimizer, loss_L1, loss_SSIM=None, epoch=0, global_step=0):
    model.train()
    total_train_loss = 0
    total_train_l1 = 0
    total_train_invert_ssim = 0
    train_samples = []

    for batch in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
        batch = batch.to(torch.bfloat16).to(device)
        batch = batch.permute(0, 4, 1, 2, 3)
        output = model(batch)["reconstructions"]
        output = output[:, :, :120, :, :]

        l1 = loss_L1(output, batch)
        invert_ssim = 1 - loss_SSIM(output, batch) if loss_SSIM else torch.tensor(0.0, device=device)
        loss = l1 + 0.1 * invert_ssim

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_l1 += l1.item()
        total_train_invert_ssim += invert_ssim.item()
        total_train_loss += loss.item()

        wandb.log({
            "Loss/train_step_l1": l1.item(),
            "Loss/train_step_invert_ssim": invert_ssim.item(),
            "Loss/train_step_total": loss.item(),
            "LR": optimizer.param_groups[0]['lr']
        }, step=global_step)

        global_step += 1

        if len(train_samples) < 4:
            idx = random.randint(0, output.shape[0] - 1)
            img_recon = output[idx, 0].detach().cpu().float().numpy()
            img_gt = batch[idx, 0].detach().cpu().float().numpy()
            depth = img_gt.shape[0] // 2
            train_samples.append((img_gt[depth], img_recon[depth]))

    avg_train_l1 = total_train_l1 / len(train_loader)
    avg_train_invert_ssim = total_train_invert_ssim / len(train_loader)
    avg_train_loss = total_train_loss / len(train_loader)

    wandb.log({
        "Loss/train_l1": avg_train_l1,
        "Loss/train_invert_ssim": avg_train_invert_ssim,
        "Loss/train_total": avg_train_loss
    }, step=global_step)

    train_images = []
    for i, (gt, recon) in enumerate(train_samples):
        train_images.append(wandb.Image(gt, caption=f"Train_GT_{i}"))
        train_images.append(wandb.Image(recon, caption=f"Train_Recon_{i}"))
    wandb.log({"Train_Images": train_images}, step=global_step)

    # ============= VALIDATION =============
    model.eval()
    total_val_loss = 0
    total_val_l1 = 0
    total_val_invert_ssim = 0
    val_samples = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            batch = batch.to(torch.bfloat16).to(device)
            batch = batch.permute(0, 4, 1, 2, 3)
            output = model(batch)[0]
            output = output[:, :, :120, :, :]

            l1 = loss_L1(output, batch)
            invert_ssim = 1 - loss_SSIM(output, batch) if loss_SSIM else torch.tensor(0.0, device=device)
            loss = l1 + 0.1 * invert_ssim

            total_val_l1 += l1.item()
            total_val_invert_ssim += invert_ssim.item()
            total_val_loss += loss.item()

            if len(val_samples) < 4:
                idx = random.randint(0, output.shape[0] - 1)
                img_recon = output[idx, 0].detach().cpu().float().numpy()
                img_gt = batch[idx, 0].detach().cpu().float().numpy()
                depth = img_gt.shape[0] // 2
                val_samples.append((img_gt[depth], img_recon[depth]))

    avg_val_l1 = total_val_l1 / len(val_loader)
    avg_val_invert_ssim = total_val_invert_ssim / len(val_loader)
    avg_val_loss = total_val_loss / len(val_loader)

    wandb.log({
        "Loss/val_l1": avg_val_l1,
        "Loss/val_invert_ssim": avg_val_invert_ssim,
        "Loss/val_total": avg_val_loss
    }, step=global_step)

    val_images = []
    for i, (gt, recon) in enumerate(val_samples):
        val_images.append(wandb.Image(gt, caption=f"Val_GT_{i}"))
        val_images.append(wandb.Image(recon, caption=f"Val_Recon_{i}"))
    wandb.log({"Val_Images": val_images}, step=global_step)

    return avg_train_loss, avg_val_loss, global_step

def training(cfg):
    wandb.login(key='9ab49432fdba1dc80b8e9b71d7faca7e8b324e3e')  # 🔁 Đổi thành API key của bạn
    wandb.init(project="O_P_C", name='cosmos_continuous', config=cfg)

    model = model_class(**config)
    log_dir = os.path.join("log", cfg["pretrained_path"].split("/")[-2] + "-Custom-Continuous-05032025")
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, "training_logs_new.log"), rotation="100 MB", level="INFO")

    # Load pretrained weights
    model_weight = torch.jit.load(cfg['pretrained_path'])
    # model_weight = torch.load('/home/jovyan/workspace/ducntm/Cosmos/finetune/ckpt/Cosmos-Tokenize1-CV8x8x8-720p-Custom-fixbug-loss-04192025/model_epoch_10.pth')
    state_dict = model_weight.state_dict()
    del model_weight
    torch.cuda.empty_cache()
    model.load_state_dict(state_dict, strict=True)
    # model.load_state_dict(model_weight, strict=True)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(torch.bfloat16).to(device)

    train_loader = get_dataloader(cfg["train"], mode="train", batch_size=8, num_workers=8)
    val_loader = get_dataloader(cfg["val"], mode="val", batch_size=8, num_workers=8)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["EPOCH"])
    loss_L1 = torch.nn.L1Loss()
    loss_SSIM = SSIM3D() if cfg.get('ssim', False) else None

    global_step = 0

    for epoch in range(cfg["EPOCH"]):
        logger.info(f"Epoch {epoch + 1}/{cfg['EPOCH']} Started")

        train_loss, val_loss, global_step = train_1_epoch(
            model, train_loader, val_loader, device,
            optimizer, loss_L1, loss_SSIM, epoch, global_step
        )

        logger.info(f"Epoch {epoch + 1} Summary - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        scheduler.step()

        # Save checkpoint
        save_dir = os.path.join(cfg["ckpt"], cfg["pretrained_path"].split("/")[-2] + "-Custom-Continuous-05032025")
        os.makedirs(save_dir, exist_ok=True)
        model_save_path = os.path.join(save_dir, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), model_save_path)
        logger.info(f"Checkpoint saved: {model_save_path}")

    wandb.finish()


if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    print(cfg)
    training(cfg)
