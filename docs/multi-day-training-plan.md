# Multi-Day Training and Testing Plan (06:30–13:50 / 16:00–23:00, two windows)

> Premise: the daily runnable periods are **06:30–13:50 and 16:00–23:00** (the remaining time is downtime or idle); training does not reduce its scale.
> Mechanism: `scripts/day_runner.py` + `runs/dayplan/tasks.json`; tasks are selected automatically inside a window,
> with a clean stop at 22:10 and continuation the next day via `--resume` (losing at most 1 epoch).

## 1. How to run

After booting up each morning, just double-click **`start_day_runner.bat`** in the project root directory.
The queue executes in the order of `tasks.json`, and progress is written to:

- `runs/dayplan/day.log` (start/end and status of each task)
- `runs/dayplan/state.json` (current task, history)
- `runs/dayplan/<task name>.log` (full output of each task)

Rules:

| Rule | Description |
|---|---|
| Window | 06:30–13:50 and 16:00–23:00; 5 minutes before each window ends, terminate the current task and save to the most recent epoch |
| Resume | All training runs carry `--resume`, and the same command continues the next day |
| Missing dependencies | If a file listed in `needs` does not exist → retry every 20 minutes, skip after 3 attempts |
| Hot reload | The queue re-reads `tasks.json` every round, so tasks can be added or removed at any time |

## 2. Daily schedule

### Day 1 (11/1, already in progress)
Overnight plan (`scripts/overnight_plan.py`):

| Stage | Content | Output |
|---|---|---|
| S1 | 384² fine-tuning for 4 rounds | `runs/whu_mt384` |
| S2 | 512² fine-tuning for 5 rounds | `runs/whu_mt512` |
| S2b | 512² continued training for 3 rounds | Same as above (8 rounds) |
| S2c | 512² final training for 2 rounds | Same as above (10 rounds) |
| S3 | Second seed, 256² for 10 rounds | `runs/whu_mt_seed7` |
| S3b | No-ASPP ablation for 8 rounds | `runs/whu_unet_edge` |
| S4 | Three evaluation suites | `results/vector_eval_512`, `vector_eval_seed7` |

### Day 2 (11/2)
| Time | Task | Description |
|---|---|---|
| 06:30–07:30 | Data health check + COCO conversion | Extract WHU-Mix, generate `data/vertex_labels` |
| 07:30–19:20 | **512² continued training to 20 rounds** (10 rounds) | About 71 min per round |
| 19:20–20:50 | 512 model pixel + vector evaluation | Update the main table of the protocol |
| 20:50–22:10 | **Vertex regression training** (if the script is ready) | Continues across days |

### Day 3 (11/3)
| Time | Task | Description |
|---|---|---|
| 06:30–16:30 | Vertex regression training (80 epochs) | True polygon labels |
| 16:30–18:30 | GCN vs. geometric baseline comparison | Decide whether the target is met |
| 18:30–22:10 | base=24 training from scratch (start) | Capacity ablation |

### Day 4 (11/4)
| Time | Task | Description |
|---|---|---|
| 06:30–14:00 | base=24 final training (20 rounds) | Capacity ablation |
| 14:00–22:10 | Second seed, 512² for 8 rounds | Ensemble/stability |

### Day 5 (11/5)
| Time | Task | Description |
|---|---|---|
| 06:30–09:30 | TTA gain (400 images) | Inference-side improvement |
| 09:30–12:00 | Cross-dataset: WHU→Massachusetts, WHU-Mix test II | Generalization |
| 12:00–16:00 | Ensemble experiment (256+512 averaging) | Comparison |
| 16:00–22:10 | Failure case screening, chart generation | Paper material |

### Day 6 (11/6)
| Time | Task | Description |
|---|---|---|
| All day | Rerun missing experiments + update documentation + first draft of thesis Chapters 3/4 | Writing |

## 3. Estimated compute and output

| Type | Estimated total duration | Output |
|---|---:|---|
| Multi-task segmentation training | ~28 h | 5 sets of weights: 256/384/512/two seeds/no-ASPP |
| Vertex regression training | ~10 h | GCN weights + comparison table |
| Evaluation and analysis | ~8 h | Pixel table, vector table, TTA, cross-domain, ensemble |
| Data preparation | ~2 h | `data/vertex_labels` |

## 4. Morning checklist (my responsibility)

1. Check `runs/overnight/status.json` or `runs/dayplan/state.json`;
2. Confirm whether the previous night's `best.pt` was updated and whether the val IoU curve is normal;
3. Connect the newly produced weights to evaluation and update `docs/vector-evaluation-protocol.md`;
4. If a task fails repeatedly, adjust the parameters and put it back into the queue;
5. Report the day's plan and yesterday's results.

## 5. Risks and responses

| Risk | Response |
|---|---|
| An epoch is running when training terminates at 22:10 | Lose at most 1 epoch (about 71 min for 512); `--resume` continues the next day |
| Power outage/shutdown | `state.json` records progress; just rerun the bat file the next day |
| Annotation offset after extracting WHU-Mix | Spot-check 20 images first; if the offset is large, use only the manually corrected subset |
| GCN does not meet the target | Downgrade it to an ablation, and return the main line to multi-task + geometry + evaluation |
| Insufficient disk space | Delete each zip immediately after extraction; about 395 GB available on drive G |

## 6. Current queue (`runs/dayplan/tasks.json`)

1. 512 continued training to 20 rounds (720 min)
2. Evaluate 512 (pixel, 50 min)
3. Evaluate 512 (vector, 40 min)
4. Vertex regression: training on true labels (600 min, requires `scripts/train_vertex.py` + data)
5. Vertex regression: comparison with the geometric baseline (120 min)
6. base=24 training from scratch for 20 rounds (540 min)
7. Second seed 512 training for 8 rounds (570 min)
8. TTA gain evaluation (180 min)
