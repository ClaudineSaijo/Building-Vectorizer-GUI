"""Train one building-extraction model on CPU."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import BuildingPatchDataset, DEFAULT_MEAN, DEFAULT_STD  # noqa: E402
from src.losses import SegEdgeLoss  # noqa: E402
from src.metrics import pixel_stats, scores_from_stats  # noqa: E402
from src.models import build_model  # noqa: E402
from src.utils import count_parameters, ensure_dir, save_json, seed_everything  # noqa: E402


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int = 0,
                pin_memory: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def forward_loss(model, criterion, x, y, e):
    out = model(x)
    if isinstance(out, tuple):
        seg, edge = out
        loss, parts = criterion(seg, y, edge, e)
        return loss, parts, seg
    loss, parts = criterion(out, y)
    return loss, parts, out


@torch.no_grad()
def validate(model, loader, criterion, device, threshold: float = 0.5, amp: bool = False) -> dict:
    model.eval()
    tp = fp = fn = 0
    loss_sum, n = 0.0, 0
    for x, y, e, _ in loader:
        x, y, e = x.to(device), y.to(device), e.to(device)
        with torch.autocast(device_type=device.type, enabled=amp):
            loss, _, seg = forward_loss(model, criterion, x, y, e)
        loss_sum += float(loss) * len(x)
        n += len(x)
        pred = (torch.sigmoid(seg.float()) >= threshold).cpu().numpy().astype(np.uint8)
        gt = y.cpu().numpy().astype(np.uint8)
        for p, g in zip(pred, gt):
            a, b, c, _ = pixel_stats(p[0], g[0])
            tp += a
            fp += b
            fn += c
    scores = scores_from_stats(tp, fp, fn)
    scores["loss"] = loss_sum / max(n, 1)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["unet", "msf", "full"], required=True)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--max-train", type=int, default=1200)
    parser.add_argument("--val-limit", type=int, default=360)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--edge-weight", type=float, default=0.35)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="auto: 有 CUDA 就用 GPU（用于 Colab/实验室机器）")
    parser.add_argument("--amp", action="store_true", help="CUDA 下启用混合精度（约 1.5-2x）")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader 进程数（GPU 上建议 4）")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    seed_everything(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    amp = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"device={device} amp={amp}")
    out_dir = ensure_dir(args.out or ROOT / "runs" / args.model)
    save_json(vars(args), out_dir / "config.json")

    proc = ROOT / "data" / "processed"
    train_ds = BuildingPatchDataset(proc / "train_images.npy", proc / "train_masks.npy", train=True, seed=args.seed)
    val_ds = BuildingPatchDataset(proc / "valid_images.npy", proc / "valid_masks.npy", train=False, seed=args.seed)
    if args.max_train and args.max_train < len(train_ds):
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(train_ds), size=args.max_train, replace=False)
        train_ds = Subset(train_ds, np.sort(idx).tolist())
    if args.val_limit and args.val_limit < len(val_ds):
        val_ds = Subset(val_ds, list(range(args.val_limit)))

    pin = device.type == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, pin)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, pin)
    model = build_model(args.model, base=args.base).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
        print("using channels_last")
    criterion = SegEdgeLoss(edge_weight=args.edge_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.05)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    print(f"model={args.model} params={count_parameters(model)/1e6:.2f}M")
    print(f"train={len(train_ds)} val={len(val_ds)} steps/epoch={len(train_loader)}")
    history = []
    best_iou = -1.0
    started = time.time()
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_iou", "val_f1",
                         "val_precision", "val_recall", "seconds"])
        for epoch in range(1, args.epochs + 1):
            model.train()
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            elif hasattr(train_ds, "dataset"):
                train_ds.dataset.set_epoch(epoch)
            running, n = 0.0, 0
            t0 = time.time()
            for step, (x, y, e, _) in enumerate(train_loader, 1):
                x, y, e = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin), e.to(device, non_blocking=pin)
                if args.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    loss, parts, _ = forward_loss(model, criterion, x, y, e)
                if amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                running += float(loss.detach()) * len(x)
                n += len(x)
                if step % 25 == 0 or step == len(train_loader):
                    print(f"  epoch {epoch:02d} step {step:03d}/{len(train_loader)} "
                          f"loss={running/n:.4f} ({time.time()-t0:.0f}s)", flush=True)
            train_loss = running / max(n, 1)
            val = validate(model, val_loader, criterion, device, amp=amp)
            scheduler.step()
            elapsed = time.time() - started
            row = {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_loss,
                "val_loss": val["loss"],
                "val_iou": val["iou"],
                "val_f1": val["f1"],
                "val_precision": val["precision"],
                "val_recall": val["recall"],
                "seconds": elapsed,
            }
            history.append(row)
            writer.writerow([row[k] for k in ["epoch", "lr", "train_loss", "val_loss", "val_iou",
                                              "val_f1", "val_precision", "val_recall", "seconds"]])
            fh.flush()
            print(f"[{args.model}] epoch {epoch:02d} train {train_loss:.4f} val_iou {val['iou']:.4f} "
                  f"val_f1 {val['f1']:.4f} ({elapsed/60:.1f} min)", flush=True)
            torch.save({"model": model.state_dict(), "epoch": epoch, "val": val,
                        "args": vars(args)}, out_dir / "last.pt")
            if val["iou"] > best_iou:
                best_iou = val["iou"]
                torch.save({"model": model.state_dict(), "epoch": epoch, "val": val,
                            "args": vars(args)}, out_dir / "best.pt")
                print(f"  new best val IoU {best_iou:.4f}", flush=True)
    save_json({"history": history, "best_iou": best_iou, "args": vars(args)}, out_dir / "history.json")
    print(f"finished in {(time.time()-started)/60:.1f} min; best val IoU={best_iou:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
