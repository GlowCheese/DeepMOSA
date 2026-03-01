import ast
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from libs.custom_logger import getLogger

from . import utils
from .wtl import show_wtl

_logger = getLogger("rtests")


class SafeTestTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        for i in range(len(node.body)):
            node.body[i] = ast.Try(
                body=[node.body[i]],
                handlers=[
                    ast.ExceptHandler(
                        type=ast.Name(id="Exception", ctx=ast.Load()), body=[ast.Pass()]
                    )
                ],
                orelse=[],
                finalbody=[],
            )
        return node


class LLMGeneratedTestAnalysis:
    def __init__(self, project: str, module: str, config_id: str) -> None:
        self.module = module
        self.project = project
        self.config_id = config_id

        self.project_path = utils.BASE_PROJECT_PATH / project
        self.module_path = self.project_path / f"{module.replace('.', '/')}.py"

        assert self.module_path.is_file()

    def _pynguin_generated_test_path(self, run_id: int):
        return Path(
            "generated_tests",
            self.project,
            self.config_id,
            f"test_{self.module.replace('.', '_')}_{run_id}.py",
        )

    def _get_run_content(self):
        report_path = utils.BASE_REPORT_PATH / self.project / self.module / self.config_id
        llm_generated = report_path / "llm" / "llm_raw_generated.py"

        content = llm_generated.read_text()
        content_split = content.split("#" * 68)

        return [content_split[i] for i in range(2, len(content_split), 2)]

    def _import_header(self, run_id: int):
        orig_generated_path = self._pynguin_generated_test_path(run_id)
        assert orig_generated_path.exists(), f"{orig_generated_path} does not exist!"
        orig_generated = orig_generated_path.read_text()
        s1 = orig_generated.find("\ndef ")
        s2 = orig_generated.find("\n@pytest")
        s = s1 if s2 == -1 or s1 < s2 else s2
        raw_imports = orig_generated[:s]
        raw_imports = self._fixup_result(raw_imports[:1000])
        try:
            imports_ast = ast.parse(raw_imports)
        except Exception:
            _logger.exception("Failed to parse raw_imports:\n%s", raw_imports)
            exit(0)
        # print(ast.dump(imports_ast, indent=2))
        fixed_imports = ""
        for stmt in imports_ast.body:
            assert isinstance(stmt, ast.Import)
            for name in stmt.names:
                if name.asname is None:
                    fixed_imports += f"import {name.name}\n"
                else:
                    assert name.asname.startswith("module_")
                    fixed_imports += f"from {name.name} import *\n"
        return fixed_imports

    def _fixup_result(self, result: str):
        if (result := result.strip()).startswith("```"):
            result = "\n".join(result.split("\n")[1:])

        try:
            node = ast.parse(result)
        except SyntaxError as e:
            line_to_rm = e.lineno
            lines = result.split("\n")
            if line_to_rm is None or line_to_rm >= len(lines):
                return self._fixup_result("\n".join(lines[:-1]))
            else:
                return self._fixup_result("\n".join(lines[:line_to_rm]))

        node = SafeTestTransformer().visit(node)
        # node = AssertionRemover().visit(node)
        return ast.unparse(node)

    def _fix_run_content(self, run_content: list[str]):
        rows = utils.read_module_statistics(self.project, self.module, self.config_id)
        assert rows

        # print([r.llm_calls for r in rows])

        j = 0
        filtered: list[str] = []

        for i in range(len(run_content)):
            while j < len(rows):
                if rows[j].llm_calls != 0:
                    break
                j += 1
                filtered.append("")
            else:
                break

            cnt = 0
            start = 0
            while True:
                sub = f"at query #{cnt + 1}"
                pos = run_content[i].find(sub, start)
                if pos == -1:
                    break
                cnt += 1
                start = pos + len(sub)

            # print(cnt)

            if cnt == rows[j].llm_calls:
                j += 1
                filtered.append(run_content[i])

        for i in range(len(filtered)):
            concat = ""

            func = ""

            def add_func():
                nonlocal func, concat
                func = self._fixup_result(func)
                if func:
                    concat += func + "\n\n"
                    func = ""

            for line in filtered[i].splitlines(keepends=False):
                if line.startswith("def "):
                    # start of a new test function
                    add_func()
                if line.startswith("def ") or line.lstrip() != line or line.strip() == "":
                    func += line + "\n"

            filtered[i] = self._import_header(i) + "\n\n" + concat

        return filtered

    def _run_test_with_coverage(self, test_path: Path):
        coverage_report_path = Path(".coverage-data/coverage.json")

        # erase previous coverage report, in advance
        subprocess.run(["coverage", "erase"])
        coverage_report_path.unlink(missing_ok=True)

        Path(".coverage-data").mkdir(parents=True, exist_ok=True)

        # fmt: off
        pwd = Path.cwd()
        command = [
            "docker", "run",     "--rm", "-t",
            "--user",            f"{os.getuid()}:{os.getgid()}",
            "--env-file",        str(pwd / ".env"),
            "-e",                f"PROJECT_NAME={self.project}",
            "-e",                "UV_CACHE_DIR=/tmp/uv-cache",
            "-e",                "PYTHONPATH=/workspace/project",

            "-e",                "RUN_COVERAGE_JSON=1",
            "-e",                "COVERAGE_FILE=/workspace/.coverage-data/.coverage",
            "-v",                f"{pwd}/.coverage-data:/workspace/.coverage-data",

            "-v",                f"{test_path}:/workspace/{test_path.name}",
            "-v",                f"{pwd}/experiments/projects/{self.project}:/workspace/project:ro",
            "-v",                f"{pwd}/.cache/project-deps:/workspace/.project-deps",
            "deepmosa-runner",
            "coverage", "run",   "--branch",
            f"--source=/workspace/project/{'/'.join(self.module.split('.')[:-1])}",
            "-m", "pytest",
            f"/workspace/{test_path.name}",
            "-q", "--no-header", "--color=no", "--disable-warnings",
            # "--no-summary"
        ]
        # fmt: on

        # result = subprocess.run(command)
        result = subprocess.run(command, stdout=subprocess.DEVNULL)

        with coverage_report_path.open("r", encoding="UTF-8") as f:
            report: dict = json.load(f)

        for path_, file_rep in report["files"].items():
            if path_.endswith(f"project/{self.module.replace('.', '/')}.py"):
                report = file_rep
                break

        result = {}
        for func, func_rep in report["functions"].items():
            result[func] = func_rep["summary"]["percent_branches_covered"]
            assert isinstance(result[func], float)

        return result

    def workflow(self):
        try:
            run_content = self._get_run_content()
        except FileNotFoundError:
            # Skipping this module as no llm call is required
            return [], []

        num_runs = len(run_content)

        assert all(c for c in run_content)

        if num_runs == 0:
            _logger.info("quitting...")
            exit(0)

        run_content = self._fix_run_content(run_content)
        if len(run_content) != utils.NUM_RUNS_PER_MODULE:
            _logger.warning("Module %s has unexpected %s runs", self.module, len(run_content))

        cov1, cov2 = [], []

        for run_id in range(len(run_content)):
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
                test_path = Path(f.name)
                f.write(run_content[run_id])

            r2 = self._run_test_with_coverage(
                self._pynguin_generated_test_path(run_id).resolve(True)
            )
            r1 = self._run_test_with_coverage(test_path)

            common_funcs = r1.keys() & r2.keys()
            test_path.unlink()

            for func in common_funcs:
                if min(r1[func], r2[func]) < 99.9:
                    cov1.append(r1[func])
                    cov2.append(r2[func])

        return cov1, cov2


o_cov1, o_cov2 = [], []
config_id = sys.argv[1]

for project in utils.find_all_projects():
    _logger.info("On project %s:", project)
    _logger.info("---")

    cov1, cov2 = [], []
    for module in utils.find_all_filtered_modules(project):
        analysis = LLMGeneratedTestAnalysis(project, module, config_id)
        result = analysis.workflow()
        cov1.extend(result[0])
        cov2.extend(result[1])

    show_wtl([("llm_tests", "deepmosa")], {"llm_tests": cov1, "deepmosa": cov2})
    _logger.info("")

    o_cov1.extend(cov1)
    o_cov2.extend(cov2)


_logger.info("")
_logger.info("OVERALL")
_logger.info("---")

show_wtl([("llm_tests", "deepmosa")], {"llm_tests": o_cov1, "deepmosa": o_cov2})
_logger.info("")
