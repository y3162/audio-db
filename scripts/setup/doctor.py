"""Report setup status as PASS, FAIL, SKIP, or NOT_TESTED.

Required failures exit non-zero. Dataset paths from .env are not printed.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.setup.assets import AssetError, check, load_manifest  # noqa: E402
from src.data.corpora_catalog import CATALOG, corpus_env_var  # noqa: E402
from src.data.constants import CORPORA_ROOT  # noqa: E402

ROLES = ("se", "asr", "official", "semambapp", "deepfilternet")
REQUIRED_ROLES = {"se", "asr", "official", "semambapp"}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def add(self, status: str, name: str, detail: str, *, required: bool) -> None:
        self.rows.append((status, name, detail, required))
        print(f"{status}\t{name}\t{detail}")

    def failed(self) -> bool:
        return any(status == "FAIL" and required for status, _name, _detail, required in self.rows)


def _python(role: str) -> Path:
    return ROOT / ".venvs" / role / "bin" / "python"


def _run(python: Path, code: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    merged = os.environ.copy()
    merged["PYTHONPATH"] = str(ROOT)
    merged["CUDA_VISIBLE_DEVICES"] = merged.get("CUDA_VISIBLE_DEVICES", "0")
    if env:
        merged.update(env)
    return subprocess.run([str(python), "-c", code], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=merged, cwd=ROOT)


def check_roles(report: Report) -> None:
    probe = (
        "import sys\n"
        "mods = ['torch', 'torchaudio']\n"
        "print(sys.version.split()[0])\n"
        "for name in mods:\n"
        "    module = __import__(name)\n"
        "    print(f'{name} {getattr(module, \"__version__\", \"?\")} {getattr(module, \"__file__\", \"?\")}')\n"
    )
    for role in ROLES:
        python = _python(role)
        required = role in REQUIRED_ROLES
        if not python.is_file():
            report.add("FAIL" if required else "NOT_TESTED", f"venv.{role}", "python missing", required=required)
            continue
        result = _run(python, probe)
        if result.returncode != 0:
            report.add("FAIL", f"venv.{role}", "import failed", required=required)
            continue
        first = result.stdout.splitlines()[0]
        report.add("PASS", f"venv.{role}", f"python {first}", required=required)


def check_torch(report: Report) -> None:
    python = _python("se")
    if not python.is_file():
        report.add("NOT_TESTED", "torch.cuda", "se environment missing", required=False)
        return
    code = (
        "import torch\n"
        "print(torch.__version__)\n"
        "print(torch.version.cuda)\n"
        "print(int(torch.cuda.is_available()))\n"
        "print(torch.__file__)\n"
    )
    result = _run(python, code)
    if result.returncode != 0:
        report.add("FAIL", "torch.cuda", "torch import failed", required=True)
        return
    version, cuda, available, origin = result.stdout.splitlines()[:4]
    report.add("PASS", "torch.version", f"{version} cuda {cuda}", required=True)
    if available != "1":
        report.add("NOT_TESTED", "torch.cuda", "CUDA is not available in this process", required=False)
        return
    occupied = _gpu_occupied()
    if occupied:
        report.add("NOT_TESTED", "torch.cuda", "GPU memory is already in use", required=False)
        return
    ran = _run(python, "import torch; torch.zeros(1, device='cuda').add_(1); print('ok')")
    if ran.returncode == 0 and ran.stdout.strip() == "ok":
        report.add("PASS", "torch.cuda", "one-element CUDA add", required=True)
    else:
        report.add("FAIL", "torch.cuda", "CUDA execution failed", required=True)


def _gpu_occupied() -> bool:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    if result.returncode != 0:
        return False
    for line in result.stdout.splitlines():
        try:
            if int(line.strip()) >= 1024:
                return True
        except ValueError:
            return True
    return False


def check_native(report: Report) -> None:
    python = _python("semambapp")
    if not python.is_file():
        report.add("FAIL", "mamba.import", "semambapp environment missing", required=True)
        return
    code = (
        "import torch\n"
        "import causal_conv1d_cuda, selective_scan_cuda\n"
        "print(causal_conv1d_cuda.__file__)\n"
        "print(selective_scan_cuda.__file__)\n"
    )
    result = _run(python, code)
    if result.returncode != 0:
        report.add("FAIL", "mamba.import", "fused extensions did not import", required=True)
    else:
        report.add("PASS", "mamba.import", "causal_conv1d_cuda and selective_scan_cuda", required=True)
        if _gpu_occupied() or _run(python, "import torch; print(int(torch.cuda.is_available()))").stdout.strip() != "1":
            report.add("NOT_TESTED", "mamba.execute", "CUDA execution was not run", required=False)
        else:
            execute = (
                "import torch\n"
                "from causal_conv1d import causal_conv1d_fn\n"
                "x = torch.randn(1, 4, 8, device='cuda')\n"
                "weight = torch.randn(4, 4, device='cuda')\n"
                "y = causal_conv1d_fn(x, weight, None)\n"
                "print(int(torch.isfinite(y).all()))\n"
            )
            ran = _run(python, execute)
            if ran.returncode == 0 and ran.stdout.strip() == "1":
                report.add("PASS", "mamba.execute", "causal_conv1d_fn finite output", required=True)
            else:
                report.add("FAIL", "mamba.execute", "fused conv execution failed", required=True)
    df = _python("deepfilternet")
    if not df.is_file():
        report.add("NOT_TESTED", "libdf.import", "deepfilternet environment missing", required=False)
    else:
        ran = _run(df, "import libdf, libdfdata; print(libdf.__file__); print(libdfdata.__file__)")
        if ran.returncode == 0:
            report.add("PASS", "libdf.import", "libdf and libdfdata from the deepfilternet environment", required=True)
        else:
            report.add("FAIL", "libdf.import", "official libdf did not import", required=True)
    se = _python("se")
    if se.is_file():
        ran = _run(
            se,
            "import json\n"
            "import os\n"
            "os.environ.pop('DF_RFFT_LIBRARY', None)\n"
            "from src.se.models.deepfilternet3.dsp import _rfft_unnormalized, rfft_runtime_record\n"
            "import torch\n"
            "record = rfft_runtime_record()\n"
            "y = _rfft_unnormalized(torch.zeros(320))\n"
            "record['finite'] = bool(torch.isfinite(y).all())\n"
            "print(json.dumps(record, sort_keys=True))\n",
            env={"DF_RFFT_REQUIRED": "1"},
        )
        detail = ran.stdout.strip().splitlines()[-1] if ran.stdout.strip() else ran.stderr.strip().splitlines()[-1:]
        detail_text = detail if isinstance(detail, str) else " ".join(detail)
        record = {}
        if ran.returncode == 0 and ran.stdout.strip():
            try:
                record = json.loads(ran.stdout.strip().splitlines()[-1])
            except json.JSONDecodeError:
                record = {}
        native = (
            ran.returncode == 0
            and record.get("kernel") == "native"
            and record.get("abi") == 1
            and record.get("finite") is True
            and isinstance(record.get("sha256"), str)
        )
        if native:
            report.add(
                "PASS",
                "df_rfft",
                f"source={record.get('source')} abi=1 sha256={record.get('sha256')} path={record.get('path')}",
                required=True,
            )
        else:
            reason = record.get("reason") or record.get("error") or detail_text or "Rust FFT did not run"
            report.add("FAIL", "df_rfft", str(reason), required=True)
    else:
        report.add("NOT_TESTED", "df_rfft", "se environment missing", required=False)


def check_assets(report: Report) -> None:
    try:
        lines = check(ROOT)
    except SystemExit as error:
        text = str(error)
        for line in text.splitlines():
            status = "FAIL" if line.startswith(("MISSING", "CONFLICT")) else "PASS"
            required = line.startswith(("MISSING weight", "CONFLICT", "MISSING source mp", "MISSING source deep", "MISSING source sem", "MISSING source clear", "MISSING source esp"))
            # Native build checkouts are optional until the extension build runs.
            if "causal_conv1d" in line or "/mamba" in line or line.endswith(" mamba"):
                required = False
            report.add(status if status == "FAIL" else "PASS", "asset", line, required=required or line.startswith("CONFLICT"))
        return
    for line in lines:
        report.add("PASS", "asset", line, required=True)


def check_models(report: Report) -> None:
    python = _python("se")
    if not python.is_file():
        report.add("NOT_TESTED", "state_dict", "se environment missing", required=False)
        return
    cases = [
        ("mp_senet", "data/checkpoints/mp-senet/g_best_vb", "", "vb"),
        ("tf_gridnet", "data/checkpoints/tf-gridnet/urgent24/exp/enh_train_enh_tfgridnet_raw/21epoch.pth", "", "urgent24"),
        ("tf_gridnet", "data/checkpoints/tf-gridnet/ms-snsd/valid.loss.best.pth", "", "ms-snsd"),
        ("deepfilternet3", "data/checkpoints/deepfilternet3/DeepFilterNet3/checkpoints/model_120.ckpt.best",
         "data/checkpoints/deepfilternet3/DeepFilterNet3/config.ini", ""),
        ("mossformergan", "data/checkpoints/mossformergan/last_best_checkpoint.pt", "", ""),
        ("semambapp", "data/checkpoints/semambapp/semambapp.pth", "", ""),
        ("tf_locoformer", "data/checkpoints/tf-locoformer/dns/valid.loss.ave_5best.pth", "", "dns"),
    ]
    code = (
        "import sys\n"
        "from src.se import load_enhancer\n"
        "name, path, config, variant = sys.argv[1:5]\n"
        "kwargs = {}\n"
        "if config: kwargs['config'] = config\n"
        "if variant: kwargs['variant'] = variant\n"
        "load_enhancer(name, path, device='cpu', **kwargs)\n"
        "print('loaded')\n"
    )
    for name, path, config, variant in cases:
        label = f"state_dict.{name}.{variant or Path(path).name}"
        if not (ROOT / path).is_file():
            report.add("NOT_TESTED", label, "checkpoint file is absent", required=False)
            continue
        result = subprocess.run(
            [str(python), "-c", code, name, path, config, variant],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT,
            env={**os.environ, "PYTHONPATH": str(ROOT), "CUDA_VISIBLE_DEVICES": ""},
        )
        if result.returncode == 0 and "loaded" in result.stdout:
            report.add("PASS", label, "strict state_dict load", required=True)
        else:
            report.add("FAIL", label, "strict state_dict load failed", required=True)


def check_links(report: Report) -> None:
    configured = bool(os.environ.get("AUDIO_DB_DATASET_ROOT", "").strip())
    for spec in CATALOG:
        if os.environ.get(corpus_env_var(spec), "").strip():
            configured = True
        dest = CORPORA_ROOT / spec.link_name
        if dest.is_symlink():
            if dest.exists():
                report.add("PASS", f"link.{spec.link_name}", "symlink resolves", required=True)
            else:
                report.add("FAIL", f"link.{spec.link_name}", "symlink is broken", required=True)
        elif dest.exists():
            report.add("FAIL", f"link.{spec.link_name}", "path exists and is not a symlink", required=True)
        elif configured:
            report.add("FAIL", f"link.{spec.link_name}", "configured dataset has no link", required=True)
        else:
            report.add("SKIP", f"link.{spec.link_name}", "no dataset path configured", required=False)


def check_api(report: Report) -> None:
    python = _python("se")
    if not python.is_file():
        report.add("NOT_TESTED", "api.se", "se environment missing", required=False)
        return
    code = (
        "from src.se import available_enhancers\n"
        "names = available_enhancers()\n"
        "assert set(names) >= {'mp_senet', 'tf_gridnet', 'deepfilternet3', 'mossformergan', 'semambapp'}\n"
        "import src.data.corpora_catalog, src.evaluation.metrics.pesq, src.asr\n"
        "print('ok')\n"
    )
    result = _run(python, code)
    if result.returncode == 0:
        report.add("PASS", "api.se", "five enhancers and data/evaluation/asr import", required=True)
    else:
        report.add("FAIL", "api.se", "common API import failed", required=True)


def main() -> None:
    report = Report()
    check_roles(report)
    check_torch(report)
    check_native(report)
    check_assets(report)
    check_models(report)
    check_links(report)
    check_api(report)
    counts: dict[str, int] = {}
    for status, _name, _detail, _required in report.rows:
        counts[status] = counts.get(status, 0) + 1
    summary = " ".join(f"{key}={counts.get(key, 0)}" for key in ("PASS", "FAIL", "SKIP", "NOT_TESTED"))
    print(f"SUMMARY\t{summary}")
    if report.failed():
        raise SystemExit(1)


if __name__ == "__main__":
    try:
        main()
    except AssetError as error:
        print(f"FAIL\tassets\t{error}")
        raise SystemExit(1) from error
