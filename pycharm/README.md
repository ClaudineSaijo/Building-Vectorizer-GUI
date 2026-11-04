# PyCharm 运行说明

1. 用 PyCharm 打开项目根目录 `D:\All4DS\BISHE`。
2. 解释器已配置为 `.venv\Scripts\python.exe`（名称为 `Python 3.13 (BISHE venv)`）。
3. 在右上角运行配置中选择 **Building Extraction Demo**。
4. 按 `Shift+F10` 运行；脚本为 `pycharm/run_demo.py`。

运行成功标志：控制台最后一行输出 `PYCHARM_DEMO_OK`，并生成：

- `pycharm_output/pycharm_run.log`
- `matlab_exchange/demo_manifest.csv`
- `matlab_exchange/demo_summary.json`
- `matlab_exchange/patches/*.png`

该脚本加载 `runs/full/best.pt`，对 8 个代表性图块做四方向翻转 TTA 推理，计算 IoU、边界 IoU、小目标召回和粘连率，供 MATLAB 继续分析。
