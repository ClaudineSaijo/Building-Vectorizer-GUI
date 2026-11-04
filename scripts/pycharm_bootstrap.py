"""Regenerate the PyCharm project files (``.idea/``) for this repository.

The repository already ships a working ``.idea`` folder.  Run this script only
when the project structure is missing or broken (e.g. the folder was copied
without ``.idea``, or PyCharm shows "no interpreter / no run configurations").

    .\\.venv\\Scripts\\python.exe scripts\\pycharm_bootstrap.py

Close PyCharm first: it rewrites these files from memory when it exits.

The script writes:
  .idea/misc.xml                     project SDK binding
  .idea/modules.xml                  module registry
  .idea/BISHE.iml                    content root, exclusions
  .idea/encodings.xml                UTF-8
  .idea/.gitignore                   IDE-local files
  .idea/runConfigurations/*.xml      GUI / self-check / ablation configurations
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IDEA = ROOT / ".idea"

RUN_CONFIGS = [
    ("Building Vectorizer GUI", "run_vectorizer.py", ""),
    ("Self Check (Vectorize)", "scripts/selfcheck_vectorize.py", ""),
    ("Ablation Benchmark", "scripts/ablation_benchmark.py", "--samples 16"),
]

MODULE_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager" inherit-compiler-output="true">
    <exclude-output />
    <content url="file://$MODULE_DIR$">
      <excludeFolder url="file://$MODULE_DIR$/.venv" />
      <excludeFolder url="file://$MODULE_DIR$/data/raw" />
      <excludeFolder url="file://$MODULE_DIR$/data/processed" />
      <excludeFolder url="file://$MODULE_DIR$/data/whu_processed" />
      <excludeFolder url="file://$MODULE_DIR$/runs" />
      <excludeFolder url="file://$MODULE_DIR$/__pycache__" />
    </content>
    <orderEntry type="jdk" jdkName="{sdk_name}" jdkType="Python SDK" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
</module>
"""

MISC_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectRootManager" version="2" project-jdk-name="{sdk_name}" project-jdk-type="Python SDK" />
</project>
"""

MODULES_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="ProjectModuleManager">
    <modules>
      <module fileurl="file://$PROJECT_DIR$/BISHE.iml" filepath="$PROJECT_DIR$/BISHE.iml" />
    </modules>
  </component>
</project>
"""

ENCODINGS_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<project version="4">
  <component name="Encoding" defaultCharsetForPropertiesFiles="UTF-8" addBOMForNewFiles="khm">
    <file url="PROJECT" charset="UTF-8" />
  </component>
</project>
"""

GITIGNORE_TEMPLATE = """workspace.xml
httpRequests/
shelf/
usage.statistics.xml
"""

RUN_TEMPLATE = """<?xml version="1.0" encoding="UTF-8"?>
<component name="ProjectRunConfigurationManager">
  <configuration default="false" name="{name}" type="PythonConfigurationType" factoryName="Python" nameIsGenerated="true">
    <module name="BISHE" />
    <option name="ENV_FILES" value="" />
    <option name="INTERPRETER_OPTIONS" value="" />
    <option name="PARENT_ENVS" value="true" />
    <envs>
      <env name="PYTHONUNBUFFERED" value="1" />
      <env name="PYTHONIOENCODING" value="utf-8" />
    </envs>
    <option name="SDK_HOME" value="{python}" />
    <option name="SDK_NAME" value="{sdk_name}" />
    <option name="WORKING_DIRECTORY" value="$PROJECT_DIR$" />
    <option name="IS_MODULE_SDK" value="false" />
    <option name="ADD_CONTENT_ROOTS" value="true" />
    <option name="ADD_SOURCE_ROOTS" value="true" />
    <option name="SCRIPT_NAME" value="$PROJECT_DIR$/{script}" />
    <option name="PARAMETERS" value="{params}" />
    <option name="SHOW_COMMAND_LINE" value="false" />
    <option name="EMULATE_TERMINAL" value="false" />
    <option name="MODULE_MODE" value="false" />
    <option name="REDIRECT_INPUT" value="false" />
    <option name="INPUT_FILE" value="" />
    <method v="2" />
  </configuration>
</component>
"""


def _existing_sdk_name() -> str | None:
    misc = IDEA / "misc.xml"
    if not misc.exists():
        return None
    match = re.search(r'project-jdk-name="([^"]+)"', misc.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def _python_version(python: Path) -> str:
    try:
        out = subprocess.run(
            [str(python), "-c", "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True, text=True, timeout=30,
        )
        return out.stdout.strip() or "3"
    except Exception:  # noqa: BLE001
        return "3"


def main() -> int:
    parser = argparse.ArgumentParser(description="生成 .idea 项目文件")
    parser.add_argument("--python", default=str(ROOT / ".venv" / "Scripts" / "python.exe"),
                        help="解释器路径，默认 .venv\\Scripts\\python.exe")
    parser.add_argument("--sdk-name", default=None,
                        help="PyCharm 中显示的 SDK 名称，默认沿用已有 venv 配置或自动推导")
    args = parser.parse_args()

    python = Path(args.python)
    if not python.exists():
        print(f"解释器不存在：{python}")
        return 1
    version = _python_version(python)
    existing = _existing_sdk_name()
    if args.sdk_name:
        sdk_name = args.sdk_name
    elif existing and "BISHE" in existing:
        sdk_name = existing
    else:
        sdk_name = f"Python {version} (BISHE venv)"

    IDEA.mkdir(exist_ok=True)
    (IDEA / "runConfigurations").mkdir(exist_ok=True)
    (IDEA / "misc.xml").write_text(MISC_TEMPLATE.format(sdk_name=sdk_name), encoding="utf-8")
    (IDEA / "modules.xml").write_text(MODULES_TEMPLATE, encoding="utf-8")
    (IDEA / "encodings.xml").write_text(ENCODINGS_TEMPLATE, encoding="utf-8")
    (IDEA / ".gitignore").write_text(GITIGNORE_TEMPLATE, encoding="utf-8")
    (ROOT / "BISHE.iml").write_text(MODULE_TEMPLATE.format(sdk_name=sdk_name), encoding="utf-8")

    for name, script, params in RUN_CONFIGS:
        path = IDEA / "runConfigurations" / f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.xml"
        path.write_text(RUN_TEMPLATE.format(name=name, script=script, params=params,
                                            python=str(python), sdk_name=sdk_name), encoding="utf-8")
        print(f"  run configuration: {name} -> {script}")

    print(f"\n完成：{IDEA}")
    print(f"  解释器: {python}")
    print(f"  SDK 名称: {sdk_name}")
    print("现在用 PyCharm 打开目录 D:\\All4DS\\BISHE 即可，运行配置在右上角下拉框里。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
