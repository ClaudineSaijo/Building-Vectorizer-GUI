# 把本目录作为 PyCharm 项目使用

## 1. 它已经是一个 PyCharm 项目

PyCharm 项目的标志是目录下的 `.idea/` 文件夹和一个 `.iml` 模块文件。本仓库已包含：

```
D:\All4DS\BISHE\
├── BISHE.iml                       # 模块：内容根、排除目录、SDK 绑定
├── .idea\
│   ├── misc.xml                    # 项目解释器（Python 3.13 (BISHE venv)）
│   ├── modules.xml                 # 模块注册
│   ├── encodings.xml               # 全项目 UTF-8
│   ├── .gitignore                  # workspace.xml 等本机文件
│   ├── inspectionProfiles\
│   └── runConfigurations\
│       ├── Building_Vectorizer_GUI.xml   # 启动可视化系统
│       ├── Self_Check_Vectorize.xml      # 无界面自检
│       ├── Ablation_Benchmark.xml        # 消融实验
│       └── Building_Extraction_Demo.xml  # 旧先导实验（无界面）
└── run_vectorizer.py               # GUI 入口
```

`.iml` 中已排除 `.venv`、`data/raw`、`data/processed`、`runs`，避免索引几十万个
数据文件；`app/`、`src/`、`scripts/` 可直接 `import`（项目根即源码根）。

## 2. 正确打开方式

1. 关闭当前 PyCharm 窗口（`File → Close Project`）；
2. `File → Open`，在文件对话框中**选中 `BISHE` 这个文件夹**，点 `OK`；
3. 若弹出 `Trust and Open Project`，选 **Trust Project**；
4. 打开后应看到：左侧 Project 面板根节点叫 `BISHE`，右下角解释器为
   **Python 3.13 (BISHE venv)**，右上角运行配置下拉框里有
   **Building Vectorizer GUI**。

> 常见错误：在 `Open` 对话框里双击进入了文件夹、选中了某个 `.py` 文件再打开，
> PyCharm 会进入“单文件/轻量”模式，看起来就不像项目。务必选中文件夹本身。
>
> 另一个常见错误：打开的是 `D:\All4DS`（上级目录），此时 `BISHE` 只是一个普通
> 子目录。请直接打开 `D:\All4DS\BISHE`。

## 3. 解释器

右下角 → `Interpreter Settings` → 应为：

```
Python 3.13 (BISHE venv)
D:\All4DS\BISHE\.venv\Scripts\python.exe
```

没有就 `Add Interpreter → Add Local Interpreter → Existing`，选上面的 `python.exe`。
不要新建虚拟环境，也不要选系统 Python。

## 4. 运行

| 配置 | 用途 |
|---|---|
| Building Vectorizer GUI | 启动可视化系统（日常用这个） |
| Self Check (Vectorize) | 矢量化 + 导出校验，输出 `SELFCHECK_OK` |
| Ablation Benchmark | 4 组配置 × 3 模型的消融实验，输出到 `results/ablation/` |

选中配置后按 `Shift+F10` 运行，`Shift+F9` 调试。工作目录已设为项目根目录，
所以脚本里的相对路径都能正确解析。

## 5. 如果项目结构丢了（比如从别处拷贝了代码）

关闭 PyCharm，然后在项目根目录执行：

```powershell
.\.venv\Scripts\python.exe scripts\pycharm_bootstrap.py
```

脚本会按当前机器重建 `.idea/misc.xml`、`modules.xml`、`encodings.xml`、`.gitignore`、
`BISHE.iml` 和三个运行配置（自动写入解释器绝对路径）。之后重新用 PyCharm 打开
`BISHE` 文件夹即可。

如需指定别的解释器：

```powershell
.\.venv\Scripts\python.exe scripts\pycharm_bootstrap.py --python D:\Python313\python.exe
```

## 6. 症状对照

| 现象 | 原因 | 处理 |
|---|---|---|
| 没有 Project 面板，只有编辑器 | 用“打开文件”方式打开了 .py | `File → Open` 选文件夹 |
| 根节点是 All4DS，BISHE 是灰文件夹 | 打开了上级目录 | 关闭项目，重新打开 BISHE |
| 右下角显示 `<No interpreter>` | SDK 未绑定或已失效 | 按第 3 节选 `.venv` 解释器 |
| 运行配置下拉框里没有 GUI | `.idea/runConfigurations` 未加载 | `File → Reload All from Disk`；仍无则跑第 5 节脚本 |
| 代码里 `import app` 报红但能运行 | 源码根未识别 | `Mark Directory as → Sources Root` 选项目根，或跑第 5 节脚本 |
| 中文注释显示乱码 | 文件编码不是 UTF-8 | `Settings → Editor → File Encodings` 全设 UTF-8；已由 `encodings.xml` 指定 |
