"""Day-window task runner: fills 06:30-22:10 with training/evaluation jobs.

The machine is only guaranteed to be on between 06:30 and 22:30, so this runner
starts a task only if it fits in the remaining window, stops cleanly at 22:10,
and resumes the same task the next morning (all training stages use --resume).

Tasks live in ``runs/dayplan/tasks.json`` so they can be edited without touching
code. Each task::

    {"name": "...", "cmd": ["scripts/train_whu.py", "..."],
     "est_min": 240, "needs": ["runs/whu_mt512/best.pt"], "resume": true}

Progress: ``runs/dayplan/state.json`` + ``runs/dayplan/day.log``.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "dayplan"
TASKS_FILE = OUT / "tasks.json"
STATE_FILE = OUT / "state.json"
LOG_FILE = OUT / "day.log"
PY = sys.executable

WINDOWS = [((6, 30), (13, 50)), ((16, 0), (23, 0))]
MIN_SLICE_MIN = 12          # don't start a task with less than this left
SLACK_MIN = 5               # stop this long before the window ends


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def now() -> datetime:
    return datetime.now()


def _at(day: datetime, hm: tuple[int, int]) -> datetime:
    return day.replace(hour=hm[0], minute=hm[1], second=0, microsecond=0)


def current_window() -> tuple[datetime, datetime] | None:
    """The window we are inside right now, or None."""
    today = now()
    for start_hm, stop_hm in WINDOWS:
        start, stop = _at(today, start_hm), _at(today, stop_hm)
        if start <= today < stop:
            return start, stop
    return None


def next_window_start() -> datetime:
    today = now()
    candidates = [_at(today, start) for start, _ in WINDOWS if _at(today, start) > today]
    if candidates:
        return min(candidates)
    return _at(today + timedelta(days=1), WINDOWS[0][0])


def in_window() -> bool:
    return current_window() is not None


def minutes_left() -> float:
    win = current_window()
    if win is None:
        return 0.0
    return (win[1] - now()).total_seconds() / 60.0


def load_json(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return default


def save_json(path: Path, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def wait_for_window() -> None:
    while not in_window():
        wait = (next_window_start() - now()).total_seconds()
        log(f"不在运行窗口（06:30-13:50 / 16:00-23:00），{wait / 60:.0f} 分钟后自动开始")
        time.sleep(min(max(wait, 60), 600))


def run_task(task: dict, state: dict) -> str:
    """Run one task; returns 'done' | 'failed' | 'interrupted'."""
    name = task["name"]
    cmd = [PY, *task["cmd"]]
    limit = min(task.get("est_min", 240) * 2, max(minutes_left() - SLACK_MIN, 1))
    log(f"开始：{name}（预计 {task.get('est_min', '?')} 分钟，本次最多 {limit:.0f} 分钟）")
    state["current"] = {"name": name, "started": time.time()}
    save_json(STATE_FILE, state)

    OUT.mkdir(parents=True, exist_ok=True)
    task_log = OUT / f"{task['name'].replace(' ', '_').replace('/', '_')[:60]}.log"
    with task_log.open("a", encoding="utf-8", errors="replace") as fh:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                                env={**os.environ, "PYTHONUNBUFFERED": "1",
                                     "PYTHONIOENCODING": "utf-8"})
        started = time.time()
        status = "done"
        while proc.poll() is None:
            if minutes_left() <= SLACK_MIN:
                log(f"到达停止时间，结束 {name}（次日续跑）")
                # On Windows .venv\Scripts\python.exe is a launcher, so terminate only kills
                # the wrapper shell while the real training process keeps running; use taskkill /T
                try:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                       capture_output=True, check=False)
                    else:
                        proc.terminate()
                    proc.wait(timeout=120)
                except Exception:  # noqa: BLE001
                    try:
                        proc.kill()
                    except Exception:
                        pass
                status = "interrupted"
                break
            time.sleep(20)
        if status != "interrupted" and proc.returncode not in (0, None):
            status = "failed"
    elapsed = (time.time() - started) / 60
    log(f"结束：{name} -> {status}，用时 {elapsed:.0f} 分钟（日志 {task_log.name}）")
    state.pop("current", None)
    state.setdefault("history", []).append(
        {"name": name, "status": status, "minutes": round(elapsed, 1),
         "at": datetime.now().strftime("%Y-%m-%d %H:%M")})
    save_json(STATE_FILE, state)
    return status


def wait_for_overnight_plan() -> None:
    """Don't start our tasks while the overnight plan is still using the CPU."""
    log_file = ROOT / "runs" / "overnight" / "plan.log"
    if not log_file.exists():
        return
    while in_window():
        try:
            text = log_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:  # noqa: BLE001
            text = ""
        if "全部阶段结束" in text or (ROOT / "runs" / "overnight" / "STOPPED").exists():
            log("过夜计划已结束（或已手动停止），可以开始今天的队列")
            return
        log("过夜计划仍在运行，5 分钟后再检查（避免与训练争抢 CPU/检查点）")
        time.sleep(300)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_json(TASKS_FILE, {"tasks": []})["tasks"]
    state = load_json(STATE_FILE, {"index": 0, "history": []})
    if not tasks:
        log(f"没有任务：请先编辑 {TASKS_FILE}")
        return 0

    wait_for_overnight_plan()

    while True:
        wait_for_window()
        # re-read the queue every iteration so it can be edited live
        tasks = load_json(TASKS_FILE, {"tasks": []})["tasks"]
        idx = state.get("index", 0)
        if idx >= len(tasks):
            log("队列已全部完成，等待窗口结束后退出")
            while in_window():
                time.sleep(60)
            return 0
        task = tasks[idx]
        if minutes_left() < MIN_SLICE_MIN:
            log("剩余时间不足，等到下一个窗口")
            time.sleep(300)
            continue

        # If a training task already has last.pt under --out but no --resume, a window restart re-runs all progress
        cmd = task.get("cmd", [])
        if "--out" in cmd and ("--resume" not in cmd):
            try:
                out_dir = cmd[cmd.index("--out") + 1]
                if (ROOT / out_dir / "last.pt").exists():
                    log(f"警告：{task['name']} 的 --out 下已有 last.pt，但没有 --resume，窗口重启会从零重跑！")
            except Exception:
                pass

        missing = [p for p in task.get("needs", []) if not (ROOT / p).exists()]
        if missing:
            retries = state.setdefault("retries", {})
            n = int(retries.get(str(idx), 0))
            if n < 3:
                retries[str(idx)] = n + 1
                save_json(STATE_FILE, state)
                log(f"{task['name']} 缺少依赖 {missing}，20 分钟后再试（第 {n + 1}/3 次）")
                time.sleep(1200)
                continue
            log(f"跳过 {task['name']}：三次尝试仍缺少 {missing}")
            state["index"] = idx + 1
            save_json(STATE_FILE, state)
            continue

        status = run_task(task, state)
        if status in ("done", "failed"):
            state["index"] = idx + 1
            save_json(STATE_FILE, state)
        # interrupted -> keep index unchanged and continue the next day with --resume

        # after one window ends, wait for the next window (06:30-13:50 / 16:00-23:00)
        if minutes_left() <= SLACK_MIN:
            log("本窗口结束，等待下一个窗口")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
