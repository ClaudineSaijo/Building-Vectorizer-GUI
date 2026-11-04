# 方案 B：WHU 数据集冲 IoU 0.85+（GPU 训练指南）

> 目标：WHU 航空影像建筑物数据集（8189 张 512²，0.3 m）上测试集 **IoU 0.85–0.89**。
> 本机只有 CPU（无 CUDA），所以训练放在 Colab / Kaggle / 实验室 GPU 上；
> 训练完把权重拷回 `runs/`，本仓库的可视化系统可直接加载。

## 1. 为什么必须换数据集

| 数据集 | 分辨率 | 公开结果 IoU | 0.85 可行性 |
|---|---|---|---|
| Massachusetts（现有） | 1 m，建筑小、边界模糊 | 0.70–0.78 | 不可行 |
| **WHU aerial** | 0.3 m，建筑大而清晰、标注干净 | **0.85–0.90** | 可行 |

这也是开题报告里写的"WHU 上 mIoU 0.88"的来源。

## 2. 前置条件

1. **GPU**：Colab 免费 T4（16 GB）足够；本地有 NVIDIA 显卡更省事。
   - 检查：`python -c "import torch; print(torch.cuda.is_available())"`
   - 本机输出是 `False`（AMD 780M 核显），所以必须用外部 GPU。
2. **数据集**：WHU 航空影像建筑物数据集（约 2.5 GB，官方来自武汉大学 GPCV 组，
   也有国内镜像）。需要能下载到本地/网盘。
3. **依赖**：在训练机上执行
   ```bash
   pip install -r requirements-whu.txt
   # 等价于：pip install segmentation-models-pytorch timm
   ```

## 3. 数据目录结构

脚本同时支持"官方划分"和"自己划分"两种结构：

```
whu_raw/
  train/{image,label}/*.tif     # 官方 train
  valid/{image,label}/*.tif     # 官方 valid（也可能是 val/validation）
  test/{image,label}/*.tif      # 官方 test
```

或者：

```
whu_raw/
  image/*.tif
  label/*.tif                   # 用 --val-frac/--test-frac 自动切分
```

> 影像与标签按**同名**配对；标签只要 0/非 0 即可，脚本会二值化。
> 大于 512 的图会被切成 512² 图块，小于 512 的会反射填充到 512²。

本机已有一份解压好的官方数据（结构已验证，已迁移到新位置）：

```
D:\All4DS\data4bishe\data4training\
  3. The cropped aerial image tiles and raster labels\
    3. The cropped image tiles and raster labels\
      train\image\*.tif  (4736)   train\label\*.tif
      val\image\*.tif    (1036)   val\label\*.tif
      test\image\*.tif   (2416)   test\label\*.tif
  whu_raster_1024\whu_raster_1024\
    r_train.zip (7.9 GB)  r_val.zip (1.6 GB)  r_test.zip (5.8 GB)   # 0.2 m / 1024² 版本
```

对应命令（注意路径含空格和中文，要加引号）：

```powershell
.\.venv\Scripts\python.exe scripts\prepare_whu.py `
  --root "D:\All4DS\data4bishe\data4training\3. The cropped aerial image tiles and raster labels\3. The cropped image tiles and raster labels" `
  --out data\whu_processed --size 512
```

> 0.2 m / 1024² 那份（`whu_raster_1024`）分辨率更高、算力约 4 倍。追求 0.85 用
> 0.3 m / 512² 就够；想冲 0.88+ 可以把 1024² 图块随机裁 512² 训练，算力与 512
> 版本相同但细节更清晰。

已在本机跑通，结果：

| split | 图块数 | 说明 |
|---|---:|---|
| train | 4736 | 512²，与官方训练区一致 |
| valid | 1036 | 官方验证区 |
| test | 2416 | 官方测试区（两块区域合计） |
| 合计 | 8188 | 官方页面写 8189，压缩包里少 1 张，属正常 |

数据集统计量（自动写入 `dataset_info.json`，训练与界面推理都会用）：
mean = `[0.4359, 0.4458, 0.4141]`，std = `[0.2160, 0.2029, 0.2169]`；
建筑像素占比：train 15.9%、valid 5.7%、test 2.5%（测试区更稀疏，虚警代价更高）。

打包产物在 `data/whu_processed/`（约 8.1 GB），训练机可以直接拷这个目录，
省去重新解压和切块。若拿到的压缩包没有 train/val/test 目录（只有扁平
`image` + `label`），加 `--official-split` 按 4736/1036/其余 切分，否则用
`--val-frac/--test-frac` 随机切。

## 4. 四步跑完

### ① 打包数据（一次性，约 10–20 分钟）

```bash
python scripts/prepare_whu.py --root /content/whu_raw --out /content/whu_processed --size 512
```

输出：`train/valid/test_{images,masks}.npy`（memmap）+ `dataset_info.json`（含均值/方差）。
**注意**：完整数据约 6.4 GB 影像 + 2.1 GB 掩膜，务必放在 Colab 的 `/content` 本地盘，
不要直接放 Google Drive（随机读会慢 10 倍以上）。

先跑 20 张验证链路可以用 `--limit 20`。

### ② 训练

```bash
python scripts/train_whu.py \
  --data /content/whu_processed \
  --arch upernet --encoder resnet34 \
  --epochs 60 --batch-size 8 --lr 1e-4 \
  --size 512 --amp --num-workers 4 \
  --out runs/whu_upernet_r34
```

关键参数：

| 参数 | 建议 | 说明 |
|---|---|---|
| `--arch` | `upernet` / `deeplabv3plus` | UPerNet 对多尺度建筑最稳 |
| `--encoder` | `resnet34`（快）、`resnet50`（更准） | ImageNet 预训练，小数据关键 |
| `--epochs` | 60（T4 约 2–3 h） | 40 epoch 也能到 0.85 左右 |
| `--batch-size` | 8（T4 16G）；显存小改 4 | |
| `--lr` | 1e-4 | AdamW + cosine |
| `--amp` | 打开 | 约 1.5–2× 加速 |
| `--num-workers` | 4 | Colab 上明显更快 |
| `--pos-weight` | 1.0 | WHU 类别较均衡；建筑占比很低时可调 2–3 |
| `--norm` | `auto` | smp 预训练模型自动用 ImageNet 均值方差；内置 U-Net 用数据集统计量 |

训练日志：`runs/whu_upernet_r34/train_log.csv`；每轮验证 IoU，自动保存 `best.pt`。
中断后加 `--resume` 继续。

### ③ 评估（阈值扫描 + 8× TTA）

```bash
python scripts/evaluate_whu.py --model whu_upernet_r34 --split test --data /content/whu_processed --tta
```

输出 `runs/whu_upernet_r34/eval_test.json`，包含每个阈值的 IoU/F1/P/R 与最优阈值。
**TTA 通常再 +0.5–1 个 IoU 点。**

### ④ 拷回本机，在界面里用

把整个 `runs/whu_upernet_r34/`（至少 `best.pt`）拷到：

```
D:\All4DS\BISHE\runs\whu_upernet_r34\
```

然后启动 GUI：

```powershell
.\.venv\Scripts\python.exe run_vectorizer.py
```

模型下拉框会自动多出 `whu_upernet_r34`（`InferenceEngine.available_models()` 会扫描
`runs/*/best.pt`）。界面会自动读取 checkpoint 里记录的：

- `arch` / `encoder`：决定网络结构；
- `mean` / `std`：WHU 的归一化统计量（不是 Massachusetts 的）；
- `size`：推理切块大小（512）。

切到该模型后照常推理、编辑顶点、导出 GeoJSON/Shapefile、跑消融对比。

## 5. 预期结果

| 配置 | 测试集 IoU | F1 | 说明 |
|---|---|---|---|
| UPerNet + ResNet-34，40 epoch，无 TTA | 0.84–0.86 | 0.91–0.92 | 最低成本达标 |
| UPerNet + ResNet-34，60 epoch + TTA | **0.86–0.88** | 0.925–0.94 | 推荐 |
| DeepLabV3+ + ResNet-50，60 epoch + TTA | 0.87–0.89 | 0.93–0.94 | 更强 |
| 加边界辅助/多尺度测试 | 0.88–0.90 | 0.94+ | 论文加分项 |

显存占用（T4 16 GB）：ResNet-34 + 512² + batch 8 约 6–8 GB；ResNet-50 约 10–12 GB。

## 6. 常见问题

| 现象 | 原因/处理 |
|---|---|
| `CUDA out of memory` | 把 `--batch-size` 降到 4；或换 `resnet34` |
| 训练很慢（>8 min/epoch） | 确认 `--amp` 已开、`--num-workers 4`、数据在本地盘不在 Drive |
| val IoU 一直 0 | 检查标签是否全 0（用 `np.unique` 看 masks）；或 `--lr` 太小 |
| IoU 卡在 0.7 左右 | 编码器没加载预训练权重：确认 `--encoder-weights imagenet` 且能联网下载 |
| 加载权重报 `Missing key(s)` | `arch`/`encoder` 与训练时不一致，检查 checkpoint 的 `args` |
| 界面里模型列表没有新模型 | 确认目录是 `runs/<名字>/best.pt` 且重启了 GUI |

## 7. 与现有代码的关系

- `src/models_seg.py`：`smp` 包装器，提供和内置 U-Net 相同的 `forward(x, return_edge)` 契约；
- `src/models.py::build_from_spec`：按 checkpoint 的 `arch` 字段选择内置 U-Net 或 smp 模型；
- `src/infer.py`：自动读取 `mean/std/size`，因此 WHU 模型和 Massachusetts 模型可以共存；
- `scripts/train_whu.py` / `evaluate_whu.py`：本指南用的训练与评估脚本；
- `scripts/prepare_whu.py`：数据打包。

> 训练侧不依赖 `app/`，可以在 Colab 上只拷 `src/` + `scripts/` + 数据即可开跑。
