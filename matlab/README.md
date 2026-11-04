# MATLAB 运行说明

前置条件：已先在 PyCharm 中成功运行 `pycharm/run_demo.py`，并关闭 PyCharm。

在项目根目录执行：

```powershell
& 'D:\MATLAB\bin\matlab.exe' -batch "run('matlab/analyze_results.m')"
```

脚本读取 `matlab_exchange/` 与 `results/model_summary.csv`，生成：

- `matlab_output/matlab_paper_results.png`：三组模型总体指标
- `matlab_output/matlab_demo_metrics.png`：8 个演示图块 IoU 与密度关系
- `matlab_output/matlab_demo_montage.png`：四类场景的预测与误差图
- `matlab_output/matlab_summary.txt`：文本摘要
- 对应的 `.fig` 文件

MATLAB 版本实测：R2020a（9.8.0.1323502），需要 Image Processing Toolbox。
