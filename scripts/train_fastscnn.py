#!/usr/bin/env python3
"""
Train Fast-SCNN via knowledge distillation from Segformer pseudo-labels.

Runs on the selfdriving-seg-lab EC2 instance (g4dn.xlarge, T4 GPU).
Also works on any machine with a CUDA GPU, or even CPU (just slower).

Expects the training_data/ directory produced by prepare_distillation_dataset.py:
    training_data/
        images/   frame_NNNNNN.png
        masks/    frame_NNNNNN.png   (0/255 binary)
        logits/   frame_NNNNNN.npy   (Segformer soft targets)
        manifest.json

Usage:
    python train_fastscnn.py --data training_data/ [--epochs 200] [--batch 16]

Outputs:
    training_data/checkpoints/fastscnn_campus_best.pth   (best val IoU)
    training_data/checkpoints/fastscnn_campus_ema.pth    (EMA weights)
    training_data/checkpoints/training_log.json          (metrics per epoch)

ML Techniques Applied:
    1. Knowledge Distillation (soft targets from Segformer logits)
    2. Heavy Data Augmentation (flip, color jitter, random crop, blur)
    3. Mixup regularization
    4. Online Hard Example Mining (OHEM) — top-30% hardest pixels
    5. Dice Loss + Weighted BCE (handles class imbalance)
    6. Cosine Annealing LR schedule
    7. Exponential Moving Average (EMA) of weights
    8. Early stopping on validation IoU
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── Import the Fast-SCNN architecture ────────────────────────────────────────
# This file should be next to train_fastscnn.py (both in scripts/)
from fast_scnn_arch import FastSCNN


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATASET WITH AUGMENTATION + MIXUP
# ═══════════════════════════════════════════════════════════════════════════════

class DistillationDataset(Dataset):
    """Loads frames, binary masks, and soft logits for distillation training."""

    def __init__(self, data_dir, frame_ids, input_size=256, augment=False):
        self.img_dir = Path(data_dir) / "images"
        self.mask_dir = Path(data_dir) / "masks"
        self.logit_dir = Path(data_dir) / "logits"
        self.frame_ids = frame_ids
        self.input_size = input_size
        self.augment = augment

        # ImageNet normalization (same as Segformer training)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.frame_ids)

    def __getitem__(self, idx):
        fid = self.frame_ids[idx]
        img = cv2.imread(str(self.img_dir / f"{fid}.png"))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_dir / f"{fid}.png"), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 127).astype(np.float32)  # 0/255 → 0.0/1.0

        # Load soft logits if available
        logit_path = self.logit_dir / f"{fid}.npy"
        has_logits = logit_path.exists()
        if has_logits:
            soft_logits = np.load(str(logit_path)).astype(np.float32)
        else:
            soft_logits = np.zeros((2, self.input_size // 4, self.input_size // 4), dtype=np.float32)

        if self.augment:
            img, mask, soft_logits = self._augment(img, mask, soft_logits)

        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - self.mean) / self.std
        img = torch.from_numpy(img.transpose(2, 0, 1))   # (3, H, W)
        mask = torch.from_numpy(mask).unsqueeze(0)         # (1, H, W)

        result = {"image": img, "mask": mask}
        if has_logits:
            result["soft_logits"] = torch.from_numpy(soft_logits)
        return result

    def _augment(self, img, mask, soft_logits):
        h, w = img.shape[:2]

        # ── Random horizontal flip (50%) ──
        if random.random() > 0.5:
            img = np.fliplr(img).copy()
            mask = np.fliplr(mask).copy()
            if soft_logits is not None:
                soft_logits = np.flip(soft_logits, axis=-1).copy()

        # ── Random crop + resize (scale 0.7 to 1.0) ──
        if random.random() > 0.3:
            scale = random.uniform(0.7, 1.0)
            ch, cw = int(h * scale), int(w * scale)
            y0 = random.randint(0, h - ch)
            x0 = random.randint(0, w - cw)
            img = cv2.resize(img[y0:y0+ch, x0:x0+cw], (w, h),
                             interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask[y0:y0+ch, x0:x0+cw], (w, h),
                              interpolation=cv2.INTER_NEAREST)
            if soft_logits is not None:
                # Logits are at 1/4 resolution
                lh, lw = soft_logits.shape[1], soft_logits.shape[2]
                ls = scale
                lch, lcw = int(lh * ls), int(lw * ls)
                ly0, lx0 = int(y0 * lh / h), int(x0 * lw / w)
                ly0 = min(ly0, lh - lch)
                lx0 = min(lx0, lw - lcw)
                cropped = soft_logits[:, ly0:ly0+lch, lx0:lx0+lcw]
                # Resize each channel
                resized = np.zeros_like(soft_logits)
                for c in range(soft_logits.shape[0]):
                    resized[c] = cv2.resize(cropped[c], (lw, lh),
                                            interpolation=cv2.INTER_LINEAR)
                soft_logits = resized

        # ── Color jitter (brightness, contrast, saturation, hue) ──
        if random.random() > 0.3:
            # Brightness
            delta = random.uniform(-30, 30)
            img = np.clip(img.astype(np.float32) + delta, 0, 255).astype(np.uint8)
            # Contrast
            alpha = random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
            # Saturation
            if random.random() > 0.5:
                hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
                hsv[:, :, 1] *= random.uniform(0.7, 1.3)
                hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
                img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        # ── Gaussian blur (20%) ──
        if random.random() > 0.8:
            ksize = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)

        return img, mask, soft_logits


def mixup_batch(batch1, batch2, alpha=0.4):
    """Mixup two batches with a random lambda from Beta(alpha, alpha)."""
    lam = np.random.beta(alpha, alpha)
    lam = max(lam, 1 - lam)  # keep lam >= 0.5 so batch1 dominates
    mixed = {}
    mixed["image"] = lam * batch1["image"] + (1 - lam) * batch2["image"]
    mixed["mask"] = lam * batch1["mask"] + (1 - lam) * batch2["mask"]
    if "soft_logits" in batch1 and "soft_logits" in batch2:
        mixed["soft_logits"] = lam * batch1["soft_logits"] + (1 - lam) * batch2["soft_logits"]
    return mixed, lam


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation — handles class imbalance."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth)


class OHEMLoss(nn.Module):
    """
    Online Hard Example Mining: focuses on the hardest fraction of pixels.
    Computes BCE on all pixels, then only backprops through the top-k hardest.
    """
    def __init__(self, hard_fraction=0.3, pos_weight=2.0):
        super().__init__()
        self.hard_fraction = hard_fraction
        self.pos_weight = torch.tensor([pos_weight])

    def forward(self, pred, target):
        pw = self.pos_weight.to(pred.device)
        # Per-pixel loss (no reduction)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, pos_weight=pw, reduction="none")
        # Flatten and take top-k hardest
        loss_flat = loss.view(-1)
        k = max(1, int(loss_flat.numel() * self.hard_fraction))
        topk, _ = torch.topk(loss_flat, k)
        return topk.mean()


class DistillationLoss(nn.Module):
    """
    Combined loss for knowledge distillation:
      - Hard loss: OHEM BCE + Dice on binary pseudo-labels
      - Soft loss: KL divergence on Segformer logits (when available)
    """
    def __init__(self, hard_weight=0.5, soft_weight=0.5, temperature=3.0,
                 drivable_ids=None):
        super().__init__()
        self.ohem = OHEMLoss(hard_fraction=0.3, pos_weight=2.0)
        self.dice = DiceLoss()
        self.hard_weight = hard_weight
        self.soft_weight = soft_weight
        self.temperature = temperature
        self.drivable_ids = drivable_ids or [2, 3, 4]

    def forward(self, pred, mask, soft_logits=None):
        """
        pred: (B, 2, H, W) raw logits from Fast-SCNN
        mask: (B, 1, H, W) binary target
        soft_logits: (B, C, h, w) teacher logits or None
        """
        # Extract the "drivable" channel logit for binary losses
        # pred has 2 classes: 0=background, 1=drivable
        pred_binary = pred[:, 1:2, :, :]  # (B, 1, H, W)

        # Hard losses on binary mask
        hard_loss = 0.5 * self.ohem(pred_binary, mask) + 0.5 * self.dice(pred_binary, mask)

        if soft_logits is None:
            return hard_loss

        # Soft distillation: KL divergence between teacher and student
        # Teacher has C classes, student has 2.
        # Collapse teacher to binary: drivable vs non-drivable
        T = self.temperature
        teacher_drivable = torch.zeros_like(soft_logits[:, 0:1, :, :])
        for did in self.drivable_ids:
            if did < soft_logits.shape[1]:
                teacher_drivable += soft_logits[:, did:did+1, :, :]
        teacher_non_drivable = soft_logits.sum(dim=1, keepdim=True) - teacher_drivable
        teacher_binary = torch.cat([teacher_non_drivable, teacher_drivable], dim=1)

        # Resize student pred to match teacher spatial dims
        student_resized = F.interpolate(pred, size=teacher_binary.shape[2:],
                                        mode="bilinear", align_corners=False)

        soft_loss = F.kl_div(
            F.log_softmax(student_resized / T, dim=1),
            F.softmax(teacher_binary / T, dim=1),
            reduction="batchmean"
        ) * (T * T)

        return self.hard_weight * hard_loss + self.soft_weight * soft_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EMA (Exponential Moving Average)
# ═══════════════════════════════════════════════════════════════════════════════

class EMA:
    """Maintains an exponential moving average of model parameters."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v

    def state_dict(self):
        return self.shadow


# ═══════════════════════════════════════════════════════════════════════════════
# 4. METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_iou(pred, target, threshold=0.5):
    """Binary IoU (Intersection over Union) for the drivable class."""
    pred_bin = (torch.sigmoid(pred[:, 1:2]) > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    if union < 1e-6:
        return 1.0
    return (intersection / union).item()


def compute_pixel_accuracy(pred, target, threshold=0.5):
    pred_bin = (torch.sigmoid(pred[:, 1:2]) > threshold).float()
    correct = (pred_bin == target).float().sum()
    total = target.numel()
    return (correct / total).item()


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss, total_iou, total_acc, n = 0, 0, 0, 0
    loader_iter = iter(loader)

    for batch in loader_iter:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        soft = batch.get("soft_logits")
        if soft is not None:
            soft = soft.to(device)

        # ── Mixup (50% of batches) ──
        if use_mixup and random.random() > 0.5:
            try:
                batch2 = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch2 = next(loader_iter)
            mixed, lam = mixup_batch(batch, batch2)
            images = mixed["image"].to(device)
            masks = mixed["mask"].to(device)
            soft = mixed.get("soft_logits")
            if soft is not None:
                soft = soft.to(device)

        outputs = model(images)
        pred = outputs[0]  # Fast-SCNN returns tuple(main, [aux])

        loss = criterion(pred, masks, soft)

        optimizer.zero_grad()
        loss.backward()
        # Gradient clipping to prevent exploding gradients on small datasets
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        total_iou += compute_iou(pred.detach(), masks)
        total_acc += compute_pixel_accuracy(pred.detach(), masks)
        n += 1

    return total_loss / max(n, 1), total_iou / max(n, 1), total_acc / max(n, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_iou, total_acc, n = 0, 0, 0, 0

    for batch in loader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        soft = batch.get("soft_logits")
        if soft is not None:
            soft = soft.to(device)

        outputs = model(images)
        pred = outputs[0]
        loss = criterion(pred, masks, soft)

        total_loss += loss.item()
        total_iou += compute_iou(pred, masks)
        total_acc += compute_pixel_accuracy(pred, masks)
        n += 1

    return total_loss / max(n, 1), total_iou / max(n, 1), total_acc / max(n, 1)


def main():
    ap = argparse.ArgumentParser(description="Train Fast-SCNN via distillation")
    ap.add_argument("--data", required=True, help="training_data/ directory")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=25,
                    help="early stopping patience (epochs without val IoU improvement)")
    ap.add_argument("--no-mixup", action="store_true", help="disable mixup")
    ap.add_argument("--no-soft", action="store_true",
                    help="disable soft distillation (use hard labels only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--weights", type=str, default="", help="Path to pre-trained weights to resume from")
    args = ap.parse_args()

    # ── Seed everything ───────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load manifest ─────────────────────────────────────────────────────
    data_dir = Path(args.data)
    manifest = json.loads((data_dir / "manifest.json").read_text())
    train_ids = manifest["train"]
    val_ids = manifest["val"]
    meta = manifest["meta"]
    input_size = meta["input_size"]
    drivable_ids = meta["drivable_ids"]

    print(f"Dataset: {len(train_ids)} train, {len(val_ids)} val, {input_size}×{input_size}")

    # ── Datasets & loaders ────────────────────────────────────────────────
    train_ds = DistillationDataset(data_dir, train_ids, input_size, augment=True)
    val_ds = DistillationDataset(data_dir, val_ids, input_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            num_workers=2, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────
    model = FastSCNN(num_classes=2, aux=True).to(device)
    if args.weights:
        model.load_state_dict(torch.load(args.weights, map_location=device))
        print(f"Loaded pre-trained weights from {args.weights}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Fast-SCNN: {total_params:,} parameters ({total_params/1e6:.2f}M)")

    # ── Optimizer + scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = DistillationLoss(
        hard_weight=0.6, soft_weight=0.4,
        temperature=3.0, drivable_ids=drivable_ids)
    if args.no_soft:
        criterion.soft_weight = 0.0
        criterion.hard_weight = 1.0
        print("Soft distillation DISABLED — using hard labels only")

    # ── EMA ────────────────────────────────────────────────────────────────
    ema = EMA(model, decay=0.999)

    # ── Training ──────────────────────────────────────────────────────────
    ckpt_dir = data_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    best_iou = 0.0
    patience_counter = 0
    log = []

    print(f"\n{'='*70}")
    print(f"Training Fast-SCNN | {args.epochs} epochs | batch {args.batch} | "
          f"lr {args.lr} | mixup={'OFF' if args.no_mixup else 'ON'}")
    print(f"{'='*70}\n")

    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_iou, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=not args.no_mixup)

        ema.update(model)
        scheduler.step()

        val_loss, val_iou, val_acc = validate(model, val_loader, criterion, device)

        lr = optimizer.param_groups[0]["lr"]
        dt = time.time() - t0

        entry = {
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "train_iou": round(train_iou, 4), "val_loss": round(val_loss, 4),
            "val_iou": round(val_iou, 4), "val_acc": round(val_acc, 4),
            "lr": round(lr, 7), "time_s": round(dt, 1)
        }
        log.append(entry)

        improved = ""
        if val_iou > best_iou:
            best_iou = val_iou
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "fastscnn_campus_best.pth")
            torch.save(ema.state_dict(), ckpt_dir / "fastscnn_campus_ema.pth")
            improved = " ★ BEST"
        else:
            patience_counter += 1

        print(f"[{epoch:3d}/{args.epochs}] "
              f"train loss={train_loss:.4f} iou={train_iou:.3f} | "
              f"val loss={val_loss:.4f} iou={val_iou:.3f} acc={val_acc:.3f} | "
              f"lr={lr:.6f} | {dt:.1f}s{improved}")

        # Save log every 10 epochs
        if epoch % 10 == 0 or improved:
            (ckpt_dir / "training_log.json").write_text(json.dumps(log, indent=2))

        # Early stopping
        if patience_counter >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
            break

    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"Training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Best val IoU: {best_iou:.4f}")
    print(f"Checkpoints saved to: {ckpt_dir}/")
    print(f"{'='*70}")

    # Final save
    (ckpt_dir / "training_log.json").write_text(json.dumps(log, indent=2))


if __name__ == "__main__":
    main()
