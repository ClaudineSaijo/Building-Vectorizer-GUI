# Using This Directory as a PyCharm Project

## 1. It Is Already a PyCharm Project

A PyCharm project is identified by the `.idea/` folder and a `.iml` module file in the directory. This repository already contains:

```
D:\All4DS\BISHE\
├── BISHE.iml                       # Module: content root, excluded directories, SDK binding
├── .idea\
│   ├── misc.xml                    # Project interpreter (Python 3.13 (BISHE venv))
│   ├── modules.xml                 # Module registration
│   ├── encodings.xml               # UTF-8 for the whole project
│   ├── .gitignore                  # Local files such as workspace.xml
│   ├── inspectionProfiles\
│   └── runConfigurations\
│       ├── Building_Vectorizer_GUI.xml   # Launches the visualization system
│       ├── Self_Check_Vectorize.xml      # Headless self-check
│       ├── Ablation_Benchmark.xml        # Ablation experiments
│       └── Building_Extraction_Demo.xml  # Old pilot experiment (headless)
└── run_vectorizer.py               # GUI entry point
```

The `.iml` already excludes `.venv`, `data/raw`, `data/processed`, and `runs` to avoid indexing
hundreds of thousands of data files; `app/`, `src/`, and `scripts/` can be imported directly
(the project root is also the source root).

## 2. How to Open It Correctly

1. Close the current PyCharm window (`File → Close Project`);
2. `File → Open`, then in the file dialog **select the `BISHE` folder itself** and click `OK`;
3. If the `Trust and Open Project` dialog appears, choose **Trust Project**;
4. After opening you should see: the root node in the Project panel on the left is named `BISHE`,
   the interpreter in the bottom-right corner is **Python 3.13 (BISHE venv)**, and the run
   configuration dropdown in the top-right corner contains **Building Vectorizer GUI**.

> Common mistake: double-clicking into the folder in the `Open` dialog and selecting a `.py`
> file before opening. PyCharm then enters "single file / lightweight" mode and does not look
> like a project. Be sure to select the folder itself.
>
> Another common mistake: opening `D:\All4DS` (the parent directory), in which case `BISHE` is
> just an ordinary subdirectory. Open `D:\All4DS\BISHE` directly.

## 3. Interpreter

Bottom-right corner → `Interpreter Settings` → it should be:

```
Python 3.13 (BISHE venv)
D:\All4DS\BISHE\.venv\Scripts\python.exe
```

If it is not there, use `Add Interpreter → Add Local Interpreter → Existing` and select the
`python.exe` above. Do not create a new virtual environment, and do not select the system Python.

## 4. Running

| Configuration | Purpose |
|---|---|
| Building Vectorizer GUI | Launches the visualization system (use this for daily work) |
| Self Check (Vectorize) | Vectorization + export validation, outputs `SELFCHECK_OK` |
| Ablation Benchmark | Ablation experiments with 4 configurations × 3 models, output to `results/ablation/` |

After selecting a configuration, press `Shift+F10` to run and `Shift+F9` to debug. The working
directory is already set to the project root, so relative paths in the scripts resolve correctly.

## 5. If the Project Structure Is Lost (e.g., the Code Was Copied from Elsewhere)

Close PyCharm, then run the following in the project root:

```powershell
.\.venv\Scripts\python.exe scripts\pycharm_bootstrap.py
```

The script rebuilds `.idea/misc.xml`, `modules.xml`, `encodings.xml`, `.gitignore`, `BISHE.iml`,
and the three run configurations for the current machine (writing the absolute interpreter path
automatically). Afterwards, reopen the `BISHE` folder in PyCharm.

To specify a different interpreter:

```powershell
.\.venv\Scripts\python.exe scripts\pycharm_bootstrap.py --python D:\Python313\python.exe
```

## 6. Symptom Reference

| Symptom | Cause | Fix |
|---|---|---|
| No Project panel, only the editor | Opened a .py via "open file" | Use `File → Open` and select the folder |
| Root node is All4DS and BISHE is a gray folder | Opened the parent directory | Close the project and reopen BISHE |
| Bottom-right corner shows `<No interpreter>` | SDK not bound or no longer valid | Select the `.venv` interpreter as in Section 3 |
| GUI missing from the run configuration dropdown | `.idea/runConfigurations` not loaded | `File → Reload All from Disk`; if it is still missing, run the script in Section 5 |
| `import app` shows red in code but runs | Source root not recognized | `Mark Directory as → Sources Root` and select the project root, or run the script in Section 5 |
| Chinese comments appear garbled | File encoding is not UTF-8 | Set everything to UTF-8 in `Settings → Editor → File Encodings`; already specified by `encodings.xml` |
