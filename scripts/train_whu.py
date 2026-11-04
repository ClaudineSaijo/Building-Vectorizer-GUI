"""Train a building segmentation model on the WHU aerial dataset.

Designed for a GPU (Colab / lab machine) but runs on CPU for smoke tests.

Example (Colab, T4)::

    python scripts/prepare_whu.py --root /content/whu_raw --out /content/whu_processed
    python scripts/train_whu.py --arch upernet --encoder resnet34 --epochs 60 \
        --batch-size 8 --lr 1e-4 --amp --num-workers 4 --out runs/whu_upernet_r34

The checkpoint stores the architecture, encoder, normalisation statistics and
tile size, so ``app/gui.py`` can load it without extra configuration.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset import DEFAULT_MEAN, DEFAULT_STD  # noqa: E402
from src.losses import SegEdgeLoss  # noqa: E402
from src.metrics import pixel_stats, scores_from_stats  # noqa: E402
from src.models import build_from_spec  # noqa: E402
from src.utils import count_parameters, ensure_dir, save_json, seed_everything  # noqa: E402
from src.whu_dataset import WhuDataset  # noqa: E402


def make_loader(ds, batch_size: int, shuffle: bool, workers: int, pin: bool):
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=pin, drop_last=False, persistent_workers=bool(workers),
    )


@torch.no_grad()
def validate(model, loader, criterion, device, amp: bool, threshold: float = 0.5) -> dict:
    model.eval()
    tp = fp = fn = 0
    loss_sum, n = 0.0, 0
    for x, y, e, _ in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        e = e.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp):
            out = model(x)
            if isinstance(out, tuple):
                seg, edge_logits = out
                loss = criterion(seg, y, edge_logits, e)[0]
            else:
                seg = out
                loss = criterion(seg, y)[0]
        loss_sum += float(loss) * len(x)
        n += len(x)
        pred = (torch.sigmoid(seg.float()) >= threshold).cpu().numpy()[:, 0].astype(np.uint8)
        gt = y.cpu().numpy()[:, 0].astype(np.uint8)
        for p, g in zip(pred, gt):
            a, b, c, _ = pixel_stats(p, g)
            tp += a
            fp += b
            fn += c
    scores = scores_from_stats(tp, fp, fn)
    scores["loss"] = loss_sum / max(n, 1)
    return scores


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "whu_processed"))
    parser.add_argument("--arch", default="upernet",
                        help="upernet / deeplabv3plus / unet_smp / fpn / manet / pspnet，"
                             "或内置 unet/msf/full（调试用）")
    parser.add_argument("--encoder", default="resnet34", help="resnet34 / resnet50 / efficientnet-b4 ...")
    parser.add_argument("--encoder-weights", default="imagenet", help="imagenet 或 none")
    parser.add_argument("--base", type=int, default=32, help="仅内置 unet/msf/full 使用")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--pos-weight", type=float, default=1.0, help="BCE 正样本权重")
    parser.add_argument("--edge-weight", type=float, default=0.5, help="边界分支损失权重")
    parser.add_argument("--edge-width", type=int, default=2, help="边界带半宽（px）")
    parser.add_argument("--occlusion-aug", type=float, default=0.0, help="遮挡增广概率（0=关闭）")
    parser.add_argument("--in-channels", type=int, default=3, choices=[3, 4], help="4 = 追加遮挡先验通道（ExG+暗区）")
    parser.add_argument("--direction-head", action="store_true", help="加 2 通道边界方向场头（BOMSC 思路）")
    parser.add_argument("--dir-weight", type=float, default=0.3, help="方向场损失权重")
    parser.add_argument("--bieb", action="store_true", help="加完整性重建分支（BIENet 思路）")
    parser.add_argument("--shadow-aug", type=float, default=0.0, help="阴影增广概率")
    parser.add_argument("--shadow-weighted-loss", action="store_true", help="阴影内建筑像素加权")
    parser.add_argument("--amodal", action="store_true", help="加可见/完整双掩膜监督（amodal）")
    parser.add_argument("--visible-weight", type=float, default=0.5, help="可见掩膜损失权重")
    parser.add_argument("--bieb-weight", type=float, default=0.2, help="完整性重建损失权重")
    parser.add_argument("--init-from", default=None,
                        help="用已有 checkpoint 初始化（strict=False，可只加载主干/解码器）")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--max-train", type=int, default=0, help="只用前 N 张（0=全部），CPU 快速基线用")
    parser.add_argument("--max-valid", type=int, default=0, help="验证集最多用 N 张（0=全部）")
    parser.add_argument("--stop-after", type=int, default=0,
                        help="本次只跑 N 个 epoch 就退出（配合 --resume 分段训练，0=跑到 --epochs）")
    parser.add_argument("--norm", default="auto", choices=["auto", "imagenet", "dataset"],
                        help="auto: smp 预训练模型用 ImageNet 统计量，内置 U-Net 用数据集统计量")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--save-every", type=int, default=200, help="每 N 步存一次断点（0=只在每轮末存）")
    parser.add_argument("--veg-boost-file", default=None, help="过采样清单（make_veg_train_subset.py 产出）")
    parser.add_argument("--veg-boost", type=int, default=1, help="清单内图块的重复倍数")
    parser.add_argument("--veg-weighted-loss", action="store_true", help="植被覆盖的建筑像素加权")
    parser.add_argument("--boundary-weighted-loss", action="store_true",
                        help="对真值边界 2px 带内的像素加权（诊断显示误差集中在边界）")
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

    data_dir = Path(args.data)
    info_path = data_dir / "dataset_info.json"
    if not info_path.exists():
        print(f"缺少 {info_path}，请先运行 scripts/prepare_whu.py")
        return 1
    info = json.loads(info_path.read_text(encoding="utf-8"))
    stats = info.get("splits", {}).get("train", {}).get("stats") or {}
    ds_mean = np.asarray(stats.get("mean", DEFAULT_MEAN), dtype=np.float32)
    ds_std = np.asarray(stats.get("std", DEFAULT_STD), dtype=np.float32)

    is_smp = args.arch.lower() not in {"unet", "msf", "full"}
    use_imagenet = args.norm == "imagenet" or (args.norm == "auto" and is_smp and args.encoder_weights != "none")
    if use_imagenet:
        from src.models_seg import IMAGENET_MEAN, IMAGENET_STD

        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
        std = np.asarray(IMAGENET_STD, dtype=np.float32)
    else:
        mean, std = ds_mean, ds_std

    boost_idx = ()
    if args.veg_boost_file:
        import json as _json

        _p = Path(args.veg_boost_file)
        if not _p.is_absolute():
            _p = ROOT / _p
        if _p.exists():
            boost_idx = tuple(t["index"] for t in _json.loads(_p.read_text(encoding="utf-8"))["tiles"])
            print(f"过采样 {len(boost_idx)} 个植被遮挡图块 x{args.veg_boost}", flush=True)
    train_ds = WhuDataset(data_dir / "train_images.npy", data_dir / "train_masks.npy",
                          mean, std, size=args.size, train=True, seed=args.seed,
                          edge_width=args.edge_width, occlusion_p=args.occlusion_aug, shadow_p=args.shadow_aug,
                          boost_indices=boost_idx, boost_factor=args.veg_boost,
                          use_prior=(args.in_channels == 4))
    val_paths = (data_dir / "valid_images.npy", data_dir / "valid_masks.npy")
    if val_paths[0].exists():
        val_ds = WhuDataset(*val_paths, mean, std, size=args.size, train=False, seed=args.seed,
                        use_prior=(args.in_channels == 4))
    else:
        val_ds = None

    # optional subsampling for CPU-only smoke runs / quick baselines
    if args.max_train and args.max_train < len(train_ds):
        rng = np.random.default_rng(args.seed)
        idx = np.sort(rng.choice(len(train_ds), size=args.max_train, replace=False))
        train_ds = torch.utils.data.Subset(train_ds, idx.tolist())
    if val_ds is not None and args.max_valid and args.max_valid < len(val_ds):
        rng = np.random.default_rng(args.seed + 1)
        idx = np.sort(rng.choice(len(val_ds), size=args.max_valid, replace=False))
        val_ds = torch.utils.data.Subset(val_ds, idx.tolist())

    pin = device.type == "cuda"
    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers, pin)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers, pin) if val_ds else None

    spec = {
        "arch": args.arch,
        "encoder": args.encoder,
        "encoder_weights": args.encoder_weights,
        "base": args.base,
        "in_channels": args.in_channels,
        "use_dir": bool(args.direction_head),
        "use_bieb": bool(args.bieb),
        "use_amodal": bool(args.amodal),
    }
    model = build_from_spec(spec).to(device)
    if args.init_from:
        init_path = Path(args.init_from)
        if init_path.exists():
            init_ck = torch.load(init_path, map_location="cpu", weights_only=False)
            state = init_ck["model"]
            if args.in_channels == 4:
                from src.models import inflate_first_conv

                state = inflate_first_conv(state, 4)
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"init from {init_path.name}: missing={len(missing)} unexpected={len(unexpected)}")
        else:
            print(f"!! init checkpoint not found: {init_path}")
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    criterion = SegEdgeLoss(bce_pos_weight=args.pos_weight, edge_weight=args.edge_weight, dice_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    out_dir = ensure_dir(args.out or ROOT / "runs" / f"whu_{args.arch}_{args.encoder.replace('-', '')}")
    ckpt_args = {
        **vars(args),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "size": args.size,
        "dataset": "whu",
        "num_train": len(train_ds),
        "num_valid": len(val_ds) if val_ds else 0,
    }
    save_json({k: v for k, v in ckpt_args.items() if k != "out"}, out_dir / "config.json")

    start_epoch = 1
    best_iou = -1.0
    if args.resume and (out_dir / "last.pt").exists():
        ck = torch.load(out_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        if "optimizer" in ck:
            optimizer.load_state_dict(ck["optimizer"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_iou = float(ck.get("best_iou", -1.0))
        print(f"resume from epoch {start_epoch - 1}")

    print(f"device={device} amp={amp} arch={args.arch} encoder={args.encoder} "
          f"params={count_parameters(model) / 1e6:.2f}M")
    print(f"train={len(train_ds)} valid={len(val_ds) if val_ds else 0} "
          f"mean={np.round(mean, 3).tolist()} std={np.round(std, 3).tolist()}")

    log_path = out_dir / "train_log.csv"
    new_log = not log_path.exists() or start_epoch == 1
    with log_path.open("a" if not new_log else "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if new_log:
            writer.writerow(["epoch", "lr", "train_loss", "val_loss", "val_iou", "val_f1", "seconds"])
        started = time.time()
        ran = 0
        for epoch in range(start_epoch, args.epochs + 1):
            if args.stop_after and ran >= args.stop_after:
                print(f"stop-after={args.stop_after} 已跑完，退出（下次用 --resume 继续）")
                break
            ran += 1
            model.train()
            if hasattr(train_ds, "set_epoch"):
                train_ds.set_epoch(epoch)
            elif hasattr(train_ds, "dataset"):
                train_ds.dataset.set_epoch(epoch)  # torch Subset wrapper
            running, n = 0.0, 0
            t0 = time.time()
            for step, (x, y, e, _) in enumerate(train_loader, 1):
                x = x.to(device, non_blocking=pin)
                y = y.to(device, non_blocking=pin)
                e = e.to(device, non_blocking=pin)
                if args.channels_last:
                    x = x.contiguous(memory_format=torch.channels_last)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type=device.type, enabled=amp):
                    out = model(x)
                    edge_logits = dir_logits = bieb_pred = None
                    if isinstance(out, tuple):
                        seg = out[0]
                        cursor = 1
                        if getattr(model, "use_edge", False):
                            edge_logits = out[cursor]; cursor += 1
                        if getattr(model, "use_dir", False):
                            dir_logits = out[cursor]; cursor += 1
                        if getattr(model, "use_bieb", False):
                            bieb_pred = out[cursor]; cursor += 1
                        vis_logits = None
                        if getattr(model, "use_amodal", False):
                            vis_logits = out[cursor]
                        loss = criterion(seg, y, edge_logits, e)[0]
                    else:
                        seg = out
                        loss = criterion(out, y)[0]
                    if dir_logits is not None:
                        from src.direction_field import boundary_direction_target, direction_loss

                        fields, bands = [], []
                        for sample in y.detach().cpu().numpy():
                            f, b = boundary_direction_target(sample[0])
                            fields.append(f); bands.append(b)
                        ft = torch.from_numpy(np.stack(fields)).float().to(device)
                        bt = torch.from_numpy(np.stack(bands)).float().to(device)
                        loss = loss + args.dir_weight * direction_loss(dir_logits.float(), ft, bt)
                    if args.boundary_weighted_loss:
                        with torch.no_grad():
                            band = torch.zeros_like(y)
                            for bi in range(y.shape[0]):
                                g = (y[bi, 0] > 0.5).to(torch.uint8).cpu().numpy()
                                if g.sum() == 0:
                                    continue
                                dt = cv2.distanceTransform(g, cv2.DIST_L2, 5)
                                inner = (dt > 0) & (dt < 2.0)
                                outer = cv2.dilate(g, np.ones((5, 5), np.uint8)) - g
                                bm = inner | (outer > 0)
                                band[bi, 0] = torch.from_numpy(bm.astype(np.float32)).to(y.device)
                            w_b = 1.0 + 3.0 * band
                        loss = loss + 0.5 * torch.nn.functional.binary_cross_entropy_with_logits(
                            seg.float(), y, weight=w_b)

                    if args.veg_weighted_loss:
                        with torch.no_grad():
                            rgb_v = (x.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                                     * np.asarray(std).reshape(1, 1, 1, -1)
                                     + np.asarray(mean).reshape(1, 1, 1, -1))
                            exg_v = 2 * rgb_v[..., 1] - rgb_v[..., 0] - rgb_v[..., 2]
                            veg_m = (exg_v > 6) & (rgb_v[..., 1] >= rgb_v[..., 0] - 2)   # includes dark green / deep green tree crowns
                            veg_t = torch.from_numpy(veg_m.astype(np.float32)[:, None]).to(device)
                        missed_v = torch.relu(y - torch.sigmoid(seg.float())) * veg_t * (y > 0.5)
                        loss = loss + 2.0 * missed_v.mean()

                    if args.shadow_weighted_loss:
                        # extra penalty for missed building pixels inside shadows
                        with torch.no_grad():
                            rgb = (x.detach().float().cpu().numpy().transpose(0, 2, 3, 1)
                                   * np.asarray(std).reshape(1, 1, 1, -1)
                                   + np.asarray(mean).reshape(1, 1, 1, -1))
                            gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
                            sh = np.stack([(g < 0.62 * np.median(g)).astype(np.float32) for g in gray])
                            sh_t = torch.from_numpy(sh[:, None]).to(device)
                        missed = torch.relu(y - torch.sigmoid(seg.float())) * sh_t * (y > 0.5)
                        loss = loss + 1.0 * missed.mean()

                    if vis_logits is not None:
                        from src.amodal import visible_target
                        import torch.nn.functional as F

                        imgs_np2 = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * std.reshape(1, 1, 1, -1)
                                    + mean.reshape(1, 1, 1, -1))
                        imgs_np2 = np.clip(imgs_np2 * 255.0, 0, 255).astype(np.uint8)
                        masks_np2 = y.detach().cpu().numpy()[:, 0]
                        vis_t = np.stack([visible_target(im, mk) for im, mk in zip(imgs_np2, masks_np2)])
                        vis_t_t = torch.from_numpy(vis_t[:, None].astype(np.float32)).to(device)
                        loss = loss + args.visible_weight * F.binary_cross_entropy_with_logits(
                            vis_logits.float(), vis_t_t)
                        # amodal must cover visible
                        loss = loss + 0.2 * torch.relu(torch.sigmoid(vis_logits.float())
                                                       - torch.sigmoid(seg.float())).mean()

                    if bieb_pred is not None:
                        from src.enhanced_image import enhanced_image

                        imgs_np = (x.detach().cpu().numpy().transpose(0, 2, 3, 1) * std.reshape(1, 1, 1, -1)
                                   + mean.reshape(1, 1, 1, -1))
                        imgs_np = np.clip(imgs_np * 255.0, 0, 255).astype(np.uint8)
                        masks_np = y.detach().cpu().numpy()[:, 0]
                        tgt = np.stack([enhanced_image(im, mk) for im, mk in zip(imgs_np, masks_np)])
                        tgt_t = torch.from_numpy(tgt.transpose(0, 3, 1, 2).astype(np.float32) / 255.0).to(device)
                        loss = loss + args.bieb_weight * torch.nn.functional.l1_loss(
                            torch.sigmoid(bieb_pred.float()), tgt_t)
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
                if args.save_every and step % args.save_every == 0:
                    torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                                "epoch": epoch, "step": step, "best_iou": best_iou,
                                "val": {"iou": best_iou}, "args": ckpt_args}, out_dir / "last.pt")
                    print(f"      [ckpt] step {step} 已保存", flush=True)
                running += float(loss.detach()) * len(x)
                n += len(x)
                if step % 50 == 0 or step == len(train_loader):
                    print(f"  epoch {epoch:02d} step {step:03d}/{len(train_loader)} "
                          f"loss={running / n:.4f} ({time.time() - t0:.0f}s)", flush=True)
            train_loss = running / max(n, 1)
            val = validate(model, val_loader, criterion, device, amp) if val_loader else {"iou": float("nan"), "f1": float("nan"), "loss": float("nan")}
            scheduler.step()
            elapsed = time.time() - started
            writer.writerow([epoch, optimizer.param_groups[0]["lr"], train_loss,
                             val.get("loss", 0.0), val.get("iou", 0.0), val.get("f1", 0.0), elapsed])
            fh.flush()
            print(f"[{out_dir.name}] epoch {epoch:02d} train {train_loss:.4f} "
                  f"val_iou {val.get('iou', float('nan')):.4f} val_f1 {val.get('f1', float('nan')):.4f} "
                  f"({elapsed / 60:.1f} min)", flush=True)

            payload = {"model": model.state_dict(), "epoch": epoch, "val": val,
                       "args": ckpt_args, "best_iou": best_iou}
            torch.save(payload, out_dir / "last.pt")
            if val.get("iou", -1) > best_iou:
                best_iou = float(val.get("iou", -1))
                payload["best_iou"] = best_iou
                torch.save(payload, out_dir / "best.pt")
                print(f"  new best val IoU {best_iou:.4f}", flush=True)

    save_json({"best_iou": best_iou, "args": ckpt_args}, out_dir / "history.json")
    print(f"finished; best val IoU={best_iou:.4f}; weights in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
