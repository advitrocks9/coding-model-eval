"""
Modal entrypoint for the eval pipeline. Runs the existing scripts/* on a
cloud GPU because the local SSH-to-GPU box is offline. The repo contents
are baked into the image; HuggingFace cache is a persistent volume so
weights download once across runs.

Usage examples:
  uv tool run modal run modal_runner.py::sanity
  uv tool run modal run modal_runner.py::run_fim_light --tag mellum_base
  uv tool run modal run modal_runner.py::run_fim_single --tag mellum_base
  uv tool run modal run modal_runner.py::run_mbpp_single --tag mellum_sft
  uv tool run modal run modal_runner.py::run_canonical_poisoning --tag mellum_sft
  uv tool run modal run modal_runner.py::run_regression_nohint --tag mellum_sft
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import modal


REPO = Path(__file__).parent
MODEL_REGISTRY = {
    "mellum_base":   "JetBrains/Mellum-4b-base",
    "mellum_sft":    "JetBrains/Mellum-4b-sft-python",
    "mellum_dpo":    "JetBrains/Mellum-4b-dpo-python",
    "ds_base":       "deepseek-ai/deepseek-coder-1.3b-base",
    "ds_instruct":   "deepseek-ai/deepseek-coder-1.3b-instruct",
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.4",
        "transformers>=4.45",
        "tokenizers>=0.20",
        "accelerate>=0.34",
        "datasets>=3.0",
        "tqdm",
        "huggingface-hub",
    )
    .add_local_dir(str(REPO / "eval"), "/repo/eval")
    .add_local_dir(str(REPO / "scripts"), "/repo/scripts")
    .add_local_dir(str(REPO / "data"), "/repo/data")
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("eval-results", create_if_missing=True)

app = modal.App("coding-model-eval")


def _resolve_model(tag: str) -> str:
    if tag.startswith("/") or tag.startswith("./"):
        return tag
    if tag in MODEL_REGISTRY:
        return MODEL_REGISTRY[tag]
    return tag


def _run(script: str, tag: str, *extra: str) -> dict:
    model_id = _resolve_model(tag)
    cmd = [
        "python", f"/repo/scripts/{script}",
        model_id, tag, *extra,
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = "/repo"
    env["HF_HOME"] = "/cache/hf"
    env["TRANSFORMERS_CACHE"] = "/cache/hf/transformers"
    env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
    print(f"\n=== {script} {tag} {' '.join(extra)} ===", flush=True)
    proc = subprocess.run(cmd, env=env, cwd="/repo")
    results_vol.commit()
    out: dict = {"script": script, "tag": tag, "args": list(extra), "returncode": proc.returncode}
    out_dir = Path("/repo/results")
    if out_dir.exists():
        latest = sorted(out_dir.glob(f"{tag}_*.jsonl"), key=lambda p: p.stat().st_mtime)
        if latest:
            out["latest_jsonl"] = latest[-1].name
            out["latest_rows"] = sum(1 for _ in latest[-1].open())
    return out


@app.function(image=image, gpu="A10G", timeout=900, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def sanity() -> dict:
    """Cheapest validation: Mellum-base on HumanEval-Infilling light (164 tasks ~5 min).
    Mellum paper claims 80.45% on HumanEval-Infilling; this should land in that ballpark."""
    return _run("run_humaneval_infilling.py", "mellum_base", "light")


@app.function(image=image, gpu="A10G", timeout=1800, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def run_fim_light(tag: str = "mellum_base") -> dict:
    return _run("run_humaneval_infilling.py", tag, "light")


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def run_fim_single(tag: str = "mellum_base") -> dict:
    return _run("run_humaneval_infilling.py", tag, "single")


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def run_mbpp_single(tag: str = "mellum_sft") -> dict:
    return _run("run_mbpp_singleturn.py", tag)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def run_canonical_poisoning(tag: str = "mellum_sft") -> dict:
    return _run("run_canonical_poisoning.py", tag)


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def run_hint_sweep(tag: str = "mellum_sft") -> dict:
    return _run("run_hint_sweep.py", tag)


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def run_fim_recovery(tag: str = "mellum_sft", variant: str = "single") -> dict:
    return _run("run_fim_recovery.py", tag, variant)


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/cache": hf_cache, "/repo/results": results_vol})
def run_regression_nohint(tag: str = "mellum_sft") -> dict:
    """Requires {tag}_singleturn.jsonl to exist in the volume already."""
    return _run("run_regression_nohint.py", tag)


@app.function(image=image, timeout=300, volumes={"/repo/results": results_vol})
def fetch_results() -> dict:
    """Returns a manifest of every results file in the volume so we can rsync down."""
    out = []
    for p in sorted(Path("/repo/results").glob("*.jsonl")):
        out.append({"name": p.name, "rows": sum(1 for _ in p.open()), "size": p.stat().st_size})
    return {"files": out}


@app.function(image=image, timeout=600, volumes={"/repo/results": results_vol})
def push_results_to_volume(files: list[tuple[str, str]]) -> dict:
    """Write a list of (name, content) tuples to the volume so an existing
    singleturn JSONL can be uploaded for run_regression_nohint."""
    written = []
    for name, content in files:
        p = Path("/repo/results") / name
        p.write_text(content)
        written.append(name)
    results_vol.commit()
    return {"written": written}


@app.function(image=image, timeout=300, volumes={"/repo/results": results_vol})
def read_results_file(name: str) -> str:
    p = Path("/repo/results") / name
    if not p.exists():
        raise FileNotFoundError(name)
    return p.read_text()
