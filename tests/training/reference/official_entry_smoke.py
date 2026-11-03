"""Launch published training programs for a few optimizer updates.

Official checkouts are not edited. Short-run settings are copied into the
output directory. SEMamba++ hardcodes its epoch count, so an external launcher
in the output directory changes only that control flow before calling main().
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FRESH = ROOT / "build" / "clean-20261104" / "sources"
OUT_DEFAULT = ROOT / "data" / "results" / "training" / "official-entry-20261104"


def _python() -> str:
    return os.environ.get("OFFICIAL_PYTHON", sys.executable)


def _source(name: str, fallback: Path) -> Path:
    fresh = FRESH / name
    if (fresh / ".git").is_dir():
        return fresh
    return fallback


def _run(cmd: list[str], log: Path, cwd: Path, env: dict | None = None, timeout: int = 1200) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    if env:
        merged.update(env)
    start = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with log.open("w") as handle:
        handle.write(f"# start {start}\n# cwd {cwd}\n# {' '.join(cmd)}\n")
        handle.flush()
        timed = subprocess.run(
            ["/usr/bin/time", "-p", *cmd],
            cwd=cwd,
            env=merged,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    text = log.read_text(errors="replace")
    real = None
    for line in text.splitlines():
        if line.startswith("real "):
            real = float(line.split()[1])
    return {"returncode": timed.returncode, "seconds": real, "log": str(log)}


def _pairs(limit: int) -> list[tuple[Path, Path]]:
    clean = ROOT / "data/corpora/VoiceBank-DEMAND/clean_trainset_28spk_wav"
    noisy = ROOT / "data/corpora/VoiceBank-DEMAND/noisy_trainset_28spk_wav"
    names = sorted(p.name for p in clean.glob("*.wav"))
    found = []
    for name in names:
        c, n = clean / name, noisy / name
        if n.is_file() and c.stat().st_size > 100_000:
            found.append((c, n))
        if len(found) == limit:
            break
    if len(found) < limit:
        raise RuntimeError(f"need {limit} VoiceBank pairs, found {len(found)}")
    return found


def _demand(limit: int) -> list[Path]:
    root = ROOT / "data/corpora/DEMAND"
    files = sorted(root.rglob("*.wav"))
    if len(files) < limit:
        raise RuntimeError(f"need {limit} DEMAND wavs, found {len(files)}")
    return files[:limit]


def run_mp(out: Path) -> dict:
    src = _source("MP-SENet", ROOT / "src/se/official/mp-senet")
    work = out / "mp-senet"
    work.mkdir(parents=True)
    pairs = _pairs(5)
    train, valid = pairs[:4], pairs[4:]
    (work / "training.txt").write_text("".join(p[0].stem + "\n" for p in train))
    (work / "validation.txt").write_text("".join(p[0].stem + "\n" for p in valid))
    config = json.loads((src / "config.json").read_text())
    published = {"batch_size": config["batch_size"], "num_workers": config["num_workers"], "segment_size": config["segment_size"]}
    config["batch_size"] = 1
    config["num_workers"] = 0
    (work / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    ckpt = work / "checkpoints"
    cmd = [
        _python(), str(src / "train.py"),
        "--config", str(work / "config.json"),
        "--input_clean_wavs_dir", str(pairs[0][0].parent),
        "--input_noisy_wavs_dir", str(pairs[0][1].parent),
        "--input_training_file", str(work / "training.txt"),
        "--input_validation_file", str(work / "validation.txt"),
        "--checkpoint_path", str(ckpt),
        "--training_epochs", "1",
        "--stdout_interval", "1",
        "--checkpoint_interval", "1000000",
        "--summary_interval", "1000000",
        "--validation_interval", "1000000",
    ]
    result = _run(cmd, work / "train.log", src, {"PYTHONPATH": str(src)})
    result.update({
        "entry": str(src / "train.py"),
        "commit": _commit(src),
        "published_config": published,
        "run_config": {"batch_size": 1, "num_workers": 0, "training_epochs": 1, "segment_size": config["segment_size"], "updates": 4},
        "deviations": [
            "batch_size 4 -> 1 so four files are four updates",
            "num_workers 4 -> 0",
            "training_epochs 400 -> 1",
            "validation and checkpoint intervals set to 1000000 so this epoch does not validate",
        ],
    })
    return result


def run_moss(out: Path) -> dict:
    src = _source("ClearerVoice-Studio", ROOT / "src/se/official/clearervoice-studio")
    se = src / "train/speech_enhancement"
    work = out / "mossformergan"
    work.mkdir(parents=True)
    pairs = _pairs(5)
    def scp(rows, path: Path) -> None:
        path.write_text("".join(f"{noisy} {clean}\n" for clean, noisy in rows))
    scp(pairs[:4], work / "train.scp")
    scp(pairs[4:], work / "valid.scp")
    cmd = [
        _python(), str(se / "train.py"),
        "--config", str(se / "config/train/MossFormerGAN_SE_16K.yaml"),
        "--network", "MossFormerGAN_SE_16K",
        "--use-cuda", "1",
        "--seed", "777",
        "--checkpoint_dir", str(work / "checkpoints"),
        "--tr-list", str(work / "train.scp"),
        "--cv-list", str(work / "valid.scp"),
        "--batch_size", "1",
        "--accu_grad", "1",
        "--effec_batch_size", "2",
        "--max_length", "2",
        "--num_workers", "0",
        "--max-epoch", "0",
        "--print_freq", "1",
        "--num-gpu", "1",
        "--init_checkpoint_path", "",
    ]
    # Published Solver calls os.path.isfile on init_checkpoint_path. The yaml
    # leaves it null, and isfile(None) raises. An empty path skips loading.
    result = _run(cmd, work / "train.log", se, {"PYTHONPATH": str(se)})
    result.update({
        "entry": str(se / "train.py"),
        "commit": _commit(src),
        "deviations": [
            "max_epoch 120 -> 0 so the loop runs one epoch (range starts at 0)",
            "four training files and one validation file instead of the published lists",
            "num_workers 4 -> 0",
            "batch_size 1, accu_grad 1, effec_batch_size 2 match the published yaml",
            "max_length 2 seconds matches the published yaml",
        ],
    })
    return result


def run_pesq(out: Path) -> dict:
    """Compare published batch_pesq with the probe's device-local replacement."""
    src = _source("ClearerVoice-Studio", ROOT / "src/se/official/clearervoice-studio")
    se = src / "train/speech_enhancement"
    work = out / "moss-pesq"
    work.mkdir(parents=True)
    script = work / "compare_pesq.py"
    script.write_text(
        "import sys\n"
        f"sys.path.insert(0, {str(se)!r})\n"
        "import torch\n"
        "from pathlib import Path\n"
        "import soundfile as sf\n"
        "from utils.misc import batch_pesq, pesq_loss\n"
        f"clean, rate = sf.read({str(_pairs(1)[0][0])!r}, dtype='float32', always_2d=False)\n"
        f"noisy, _ = sf.read({str(_pairs(1)[0][1])!r}, dtype='float32', always_2d=False)\n"
        "n = min(len(clean), len(noisy), 16000)\n"
        "clean, noisy = clean[:n], noisy[:n]\n"
        "official = batch_pesq([clean], [noisy]).detach().cpu()\n"
        "replaced = torch.tensor([(pesq_loss(clean, noisy) - 1) / 3.5], dtype=torch.float32)\n"
        "diff = float((official - replaced).abs().max())\n"
        "print('pesq_diff', diff)\n"
        "print('official', official.tolist(), 'replaced', replaced.tolist())\n"
        "raise SystemExit(0 if diff == 0 else 2)\n"
    )
    result = _run([_python(), str(script)], work / "pesq.log", work)
    result["note"] = "published batch_pesq hardcodes CUDA; the probe calls pesq_loss and the same (score-1)/3.5 normalization on the step device"
    return result


def run_tf(out: Path) -> dict:
    work = out / "tf-gridnet"
    work.mkdir(parents=True)
    esp = _source("espnet", ROOT / "src/se/official/espnet")
    entry = esp / "espnet2/bin/enh_train.py"
    import_log = work / "enh_train_import.log"
    imported = _run([_python(), "-c", f"import sys; sys.path.insert(0, {str(esp)!r}); import espnet2.bin.enh_train"], import_log, ROOT)
    pairs = _pairs(2)
    payload = work / "input.pt"
    import torch
    clean = []
    noisy = []
    items = []
    for index, (cpath, npath) in enumerate(pairs):
        import soundfile as sf
        c, rate = sf.read(cpath, dtype="float32", always_2d=False)
        n, _ = sf.read(npath, dtype="float32", always_2d=False)
        length = min(len(c), len(n), 2048)
        clean.append(torch.from_numpy(c[:length].copy()))
        noisy.append(torch.from_numpy(n[:length].copy()))
        items.append({"case_id": cpath.stem, "sample_rate": rate, "length": length})
    torch.save({"clean": clean, "noisy": noisy, "items": items}, payload)
    probe = ROOT / "tests/training/reference/probe_tf_gridnet_official_stft.py"
    ran = _run(
        [_python(), str(probe), "--output", str(work / "official-modules.json"), "--input", str(payload), "--device", "cuda:0", "--steps", "2"],
        work / "modules.log",
        ROOT,
        {"PYTHONPATH": str(ROOT), "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
    )
    return {
        "entry_present": entry.is_file(),
        "entry": str(entry),
        "commit": _commit(esp),
        "enh_train_import": imported,
        "official_module_steps": ran,
        "classification": "espnet2.bin.enh_train was not executed as a training process. The import result is separate from the two updates that call the checked-out TFGridNet, STFT encoder/decoder, and MultiResL1SpecLoss.",
    }


def run_df(out: Path) -> dict:
    src = _source("DeepFilterNet", ROOT / "src/se/official/deepfilternet")
    df_root = src / "DeepFilterNet"
    work = out / "deepfilternet3"
    work.mkdir(parents=True)
    py = os.environ.get("DF_PYTHON", str(ROOT / ".venvs/clean-20261104-df/bin/python"))
    if not Path(py).is_file():
        return {"skipped": True, "reason": f"DF python missing: {py}"}
    speech = _pairs(2)
    noise = _demand(2)
    rir = work / "rir.wav"
    import numpy as np
    import soundfile as sf
    impulse = np.zeros(4800, dtype="float32")
    impulse[0] = 1.0
    sf.write(rir, impulse, 48000)
    lists = {}
    for kind, rows in {
        "speech": [p[0] for p in speech],
        "noise": noise,
        "rir": [rir],
    }.items():
        listing = work / f"{kind}.txt"
        listing.write_text("".join(str(path) + "\n" for path in rows))
        lists[kind] = listing
    data_dir = work / "hdf5"
    data_dir.mkdir()
    prep = []
    env = {"PYTHONPATH": str(df_root)}
    for kind in ("speech", "noise", "rir"):
        prep.append(_run(
            [py, str(df_root / "df/scripts/prepare_data.py"), "--sr", "48000", "--num_workers", "1", kind, str(lists[kind]), str(data_dir / f"{kind}.hdf5")],
            work / f"prepare-{kind}.log",
            df_root,
            env,
        ))
    config_src = ROOT / "data/results/training/final-h100-20261102-cpu/df-config.ini"
    text = config_src.read_text()
    replacements = {
        "batch_size = 64": "batch_size = 1",
        "batch_size_eval = 64": "batch_size_eval = 1",
        "num_workers = 16": "num_workers = 0",
        "max_sample_len_s = 3.0": "max_sample_len_s = 1.0",
        "max_epochs = 120": "max_epochs = 1",
        "num_prefetch_batches = 8": "num_prefetch_batches = 1",
        "p_reverb = 0.1": "p_reverb = 0.0",
        "batch_size_scheduling = 0/16,2/24,5/32,10/64,20/128,40/9999": "batch_size_scheduling = 0/1",
        "log_freq = 100": "log_freq = 1",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"config anchor missing: {old}")
        text = text.replace(old, new, 1)
    base = work / "run"
    base.mkdir()
    (base / "config.ini").write_text(text)
    dataset = {
        "train": [["speech.hdf5", 1.0], ["noise.hdf5", 1.0], ["rir.hdf5", 1.0]],
        "valid": [["speech.hdf5", 1.0], ["noise.hdf5", 1.0], ["rir.hdf5", 1.0]],
        "test": [["speech.hdf5", 1.0], ["noise.hdf5", 1.0], ["rir.hdf5", 1.0]],
    }
    (work / "dataset.cfg").write_text(json.dumps(dataset, indent=2) + "\n")
    trained = _run(
        [py, str(df_root / "df/train.py"), str(work / "dataset.cfg"), str(data_dir), str(base), "--no-resume"],
        work / "train.log",
        df_root,
        env,
        timeout=1800,
    )
    return {
        "entry": str(df_root / "df/train.py"),
        "commit": _commit(src),
        "prepare": prep,
        "train": trained,
        "rir": "synthetic unit impulse, p_reverb 0",
        "deviations": list(replacements.values()),
    }


def run_semamba(out: Path) -> dict:
    src = _source("SEMambapp", ROOT / "src/se/official/semambapp")
    work = out / "semambapp"
    work.mkdir(parents=True)
    import yaml
    config = yaml.safe_load((src / "config.yaml").read_text())
    published = {
        "batch_size": config["training_cfg"]["batch_size"],
        "num_workers": config["env_setting"]["num_workers"],
        "segment_size": config["training_cfg"]["segment_size"],
    }
    config["training_cfg"]["batch_size"] = 1
    config["env_setting"]["num_workers"] = 0
    config["training_cfg"]["segment_size"] = 16000
    (work / "semambapp_cfg.yaml").write_text(yaml.safe_dump(config))
    clean, noisy = _pairs(2)[0]
    noise = _demand(1)[0]
    rir = work / "rir.wav"
    import numpy as np
    import soundfile as sf
    impulse = np.zeros(1600, dtype="float32")
    impulse[0] = 1.0
    sf.write(rir, impulse, 16000)
    (work / "clean.json").write_text(json.dumps([str(clean)]))
    (work / "noise.json").write_text(json.dumps([str(noise)]))
    (work / "rir.json").write_text(json.dumps([str(rir)]))
    (work / "valid_clean.json").write_text("[]")
    (work / "valid_degraded.json").write_text("[]")
    launcher = work / "launch.py"
    launcher.write_text(
        "import importlib.util, os, sys, types\n"
        f"sys.path.insert(0, {str(src)!r})\n"
        "os.chdir({cwd!r})\n"
        "# dataset.py imports utils.util, but the pinned tree only has utils.py.\n"
        "# Register both before import. Official files are not edited.\n"
        f"_utils_path = {str(src / 'utils.py')!r}\n"
        "_spec = importlib.util.spec_from_file_location('_semamba_utils_file', _utils_path)\n"
        "_file = importlib.util.module_from_spec(_spec)\n"
        "_spec.loader.exec_module(_file)\n"
        "_pkg = types.ModuleType('utils')\n"
        "_pkg.__file__ = _utils_path\n"
        "_pkg.__path__ = []\n"
        "_pkg.__dict__.update(_file.__dict__)\n"
        "_util = types.ModuleType('utils.util')\n"
        "_util.load_config = _file.load_config\n"
        "sys.modules['utils'] = _pkg\n"
        "sys.modules['utils.util'] = _util\n"
        "import inspect, train\n"
        "source = inspect.getsource(train.main)\n"
        "source = source.replace('\"training_epochs\": 100', '\"training_epochs\": 1', 1)\n"
        "source = source.replace('\"stdout_interval\": 1250', '\"stdout_interval\": 1', 1)\n"
        "source = source.replace('\"checkpoint_interval\": 5000', '\"checkpoint_interval\": 10**9', 1)\n"
        "source = source.replace('\"summary_interval\": 1250', '\"summary_interval\": 10**9', 1)\n"
        "source = source.replace('\"validation_interval\": 5000', '\"validation_interval\": 10**9', 1)\n"
        "_exp = {exp!r}\n"
        "source = source.replace('\"exp_path\": \"exp\"', '\"exp_path\": ' + repr(_exp), 1)\n"
        "source = source.replace('\"experiment_name\": \"train_semambapp\"', '\"experiment_name\": \"smoke\"', 1)\n"
        "source = source.replace(\"'semambapp_cfg.yaml'\", \"'semambapp_cfg.yaml'\", 1)\n"
        "source = source.replace(\"'data/train_speech.json'\", \"'clean.json'\", 1)\n"
        "source = source.replace(\"'data/train_noise.json'\", \"'noise.json'\", 1)\n"
        "source = source.replace(\"'data/train_rir.json'\", \"'rir.json'\", 1)\n"
        "source = source.replace(\"'data/val_clean.json'\", \"'valid_clean.json'\", 1)\n"
        "source = source.replace(\"'data/val_degraded.json'\", \"'valid_degraded.json'\", 1)\n"
        "exec(source, train.__dict__)\n"
        "train.main()\n".format(cwd=str(work), exp=str(work / "exp"))
    )
    env = {
        "PYTHONPATH": str(src),
        "WANDB_MODE": "offline",
        "WANDB_DIR": str(work / "wandb"),
    }
    result = _run([_python(), str(launcher)], work / "train.log", work, env, timeout=1800)
    result.update({
        "entry": str(src / "train.py"),
        "commit": _commit(src),
        "published_config": published,
        "launcher": "external EasyDict wrapper sets one epoch and disables validation; official train.py is not edited. The launcher also registers utils.util because dataset.py imports it and the pinned tree only contains utils.py. That import is unused by the dataset.",
        "deviations": [
            "batch_size 8 -> 1",
            "num_workers 32 -> 0",
            "segment_size 48000 -> 16000",
            "training_epochs 100 -> 1 via external launcher",
            "one clean file, one DEMAND noise, synthetic impulse as RIR",
            "WANDB_MODE=offline",
        ],
    })
    return result


def _commit(path: Path) -> str:
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT_DEFAULT)
    parser.add_argument("--only", nargs="+", choices=("mp", "moss", "pesq", "tf", "df", "semamba"), default=["mp", "moss", "pesq", "tf"])
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite {args.output}")
    args.output.mkdir(parents=True)
    runners = {"mp": run_mp, "moss": run_moss, "pesq": run_pesq, "tf": run_tf, "df": run_df, "semamba": run_semamba}
    report = {"python": _python(), "only": args.only, "models": {}}
    for name in args.only:
        print("RUN", name, flush=True)
        try:
            report["models"][name] = runners[name](args.output)
        except subprocess.TimeoutExpired as exc:
            report["models"][name] = {"timeout": True, "error": str(exc)}
        except Exception as exc:
            report["models"][name] = {"error": f"{type(exc).__name__}: {exc}"}
        print("DONE", name, report["models"][name].get("returncode", report["models"][name].get("error")), flush=True)
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(args.output / "report.json")


if __name__ == "__main__":
    main()
