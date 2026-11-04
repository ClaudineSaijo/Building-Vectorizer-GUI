"""Unattended overnight training plan (CPU).

Runs a fixed sequence of stages, one after another, logging everything so the
results can be reviewed in the morning:

    S0  wait for the currently running multi-task job (runs/whu_mt_edge) to finish
    S1  fine-tune that model at 512x512 for 6 epochs      (~5.5 h)
    S2  6 more epochs at 512x512, lower LR                (~5.5 h)
    S3  a second seed at 256x256, for stability/ensemble   (~2 h)
    S4  evaluation suite (mask IoU, edge F1, vector table)  (~1 h)

Usage::

    .\\.venv\\Scripts\\python.exe scripts\\overnight_plan.py

Progress: ``runs/overnight/status.json`` and ``runs/overnight/plan.log``.
Every stage is independent: a failure is logged and the next stage still runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "overnight"
LOG = OUT / "plan.log"
STATUS = OUT / "status.json"
PY = sys.executable

TRAIN = "scripts/train_whu.py"
EVAL_PIX = "scripts/evaluate_whu.py"
EVAL_VEC = "scripts/evaluate_vectors_whu.py"

STAGES = [
    {
        "name": "S1 384 微调 4 epoch",
        "cmd": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                "--size", "384", "--epochs", "4", "--batch-size", "4", "--lr", "5e-5",
                "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                "--threads", "12", "--num-workers", "0",
                "--init-from", "runs/whu_mt_edge/best.pt", "--out", "runs/whu_mt384"],
        "minutes": 165,
        "retry": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                  "--size", "384", "--epochs", "4", "--batch-size", "2", "--lr", "5e-5",
                  "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                  "--threads", "12", "--num-workers", "0",
                  "--init-from", "runs/whu_mt_edge/best.pt", "--out", "runs/whu_mt384"],
    },
    {
        "name": "S2 512 微调 5 epoch",
        "cmd": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                "--size", "512", "--epochs", "5", "--batch-size", "4", "--lr", "3e-5",
                "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                "--threads", "12", "--num-workers", "0",
                "--init-from", "runs/whu_mt384/best.pt", "--out", "runs/whu_mt512"],
        "minutes": 360,
        "retry": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                  "--size", "512", "--epochs", "5", "--batch-size", "2", "--lr", "3e-5",
                  "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                  "--threads", "12", "--num-workers", "0",
                  "--init-from", "runs/whu_mt384/best.pt", "--out", "runs/whu_mt512"],
    },
    {
        "name": "S2b 512 续训 3 epoch",
        "cmd": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                "--size", "512", "--epochs", "8", "--batch-size", "4", "--lr", "1e-5",
                "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                "--threads", "12", "--num-workers", "0",
                "--resume", "--out", "runs/whu_mt512"],
        "minutes": 220,
        "retry": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                  "--size", "512", "--epochs", "8", "--batch-size", "2", "--lr", "1e-5",
                  "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                  "--threads", "12", "--num-workers", "0",
                  "--resume", "--out", "runs/whu_mt512"],
    },
    {
        "name": "S2c 512 收尾 2 epoch",
        "cmd": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                "--size", "512", "--epochs", "10", "--batch-size", "4", "--lr", "5e-6",
                "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                "--threads", "12", "--num-workers", "0",
                "--resume", "--out", "runs/whu_mt512"],
        "minutes": 150,
        "retry": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                  "--size", "512", "--epochs", "10", "--batch-size", "2", "--lr", "5e-6",
                  "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                  "--threads", "12", "--num-workers", "0",
                  "--resume", "--out", "runs/whu_mt512"],
    },
    {
        "name": "S3 第二种子 256",
        "cmd": [TRAIN, "--data", "data/whu_processed", "--arch", "full", "--base", "16",
                "--size", "256", "--epochs", "10", "--batch-size", "8", "--lr", "3e-4",
                "--pos-weight", "1.5", "--edge-weight", "0.5", "--seed", "7",
                "--device", "cpu", "--threads", "12", "--num-workers", "0",
                "--init-from", "runs/whu_cpu_unet/best.pt", "--out", "runs/whu_mt_seed7"],
        "minutes": 130,
    },
    {
        "name": "S3b 消融：无 ASPP 的多任务 10 epoch",
        "cmd": [TRAIN, "--data", "data/whu_processed", "--arch", "unet_edge", "--base", "16",
                "--size", "256", "--epochs", "8", "--batch-size", "8", "--lr", "3e-4",
                "--pos-weight", "1.5", "--edge-weight", "0.5", "--device", "cpu",
                "--threads", "12", "--num-workers", "0",
                "--init-from", "runs/whu_cpu_unet/best.pt", "--out", "runs/whu_unet_edge"],
        "minutes": 105,
    },
    {
        "name": "S4a 评估 512 模型（像素）",
        "cmd": [EVAL_PIX, "--model", "whu_mt512", "--split", "test"],
        "minutes": 45,
    },
    {
        "name": "S4b 评估 512 模型（矢量 + 边界 F1）",
        "cmd": [EVAL_VEC, "--model", "whu_mt512", "--use-edge", "--samples", "80",
                "--threshold", "0.75", "--out", "results/vector_eval_512"],
        "minutes": 20,
    },
    {
        "name": "S4c 评估第二种子（矢量 + 边界 F1）",
        "cmd": [EVAL_VEC, "--model", "whu_mt_seed7", "--use-edge", "--samples", "80",
                "--threshold", "0.75", "--out", "results/vector_eval_seed7"],
        "minutes": 20,
    },
]


def log(msg: str) -> None:
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_status(state: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_current_training(max_minutes: int = 200) -> None:
    """Wait until runs/whu_mt_edge/last.pt reports epoch >= 12 (or timeout)."""
    last = ROOT / "runs" / "whu_mt_edge" / "last.pt"
    deadline = time.time() + max_minutes * 60
    log("S0 等待当前多任务训练结束（最多 200 分钟）")
    while time.time() < deadline:
        try:
            import torch

            if last.exists():
                ck = torch.load(last, map_location="cpu", weights_only=False)
                epoch = int(ck.get("epoch", 0))
                if epoch >= 12:
                    log(f"S0 当前训练已完成到 epoch {epoch}")
                    return
        except Exception as exc:  # noqa: BLE001
            log(f"S0 读取 checkpoint 失败（可能正在写入）：{exc}")
        time.sleep(60)
    log("S0 等待超时，继续执行后续阶段")


def run_stage(index: int, stage: dict, state: dict) -> None:
    name = stage["name"]
    cmd = [PY, *stage["cmd"]]
    log(f"{name} 开始（预计 {stage['minutes']} 分钟）")
    state["stages"][index] = {"name": name, "status": "running", "start": time.time()}
    write_status(state)
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    stage_log = OUT / f"stage{index + 1}.log"
    ok = _run(cmd, stage_log)
    elapsed = (time.time() - started) / 60
    if not ok and "retry" in stage:
        log(f"{name} 失败，用更保守的参数重试一次")
        cmd = [PY, *stage["retry"]]
        ok = _run(cmd, stage_log)
        elapsed = (time.time() - started) / 60
    log(f"{name} {'完成' if ok else '失败'}，用时 {elapsed:.0f} 分钟（日志 {stage_log.name}）")
    state["stages"][index].update({"status": "done" if ok else "failed",
                                   "minutes": round(elapsed, 1)})
    write_status(state)


def _run(cmd: list[str], stage_log: Path) -> bool:
    try:
        with stage_log.open("a", encoding="utf-8", errors="replace") as fh:
            proc = subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                  env={**os.environ, "PYTHONUNBUFFERED": "1",
                                       "PYTHONIOENCODING": "utf-8"})
        return proc.returncode == 0
    except Exception as exc:  # noqa: BLE001
        log(f"启动失败：{exc}")
        return False


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    log("=" * 60)
    log("隔夜训练计划启动")
    state = {"started": time.time(), "stages": {}}
    write_status(state)
    wait_for_current_training()
    for i, stage in enumerate(STAGES):
        run_stage(i, stage, state)
    state["finished"] = time.time()
    write_status(state)
    log("全部阶段结束，查看 runs/overnight/status.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
