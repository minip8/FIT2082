"""Shared helpers for writing run results and measuring memory.

The result files under `results/` follow one schema regardless of what wrote
them -- `notebooks/compare.ipynb`, or a script under `scripts/`:

    {commit, dataset, device, split, transform,
     models: {name: {x_name, params, timings, results: {split: {metric: curve}}}}}

so that a plot reading one file does not care which model came from where.
"""

import json
import resource
import subprocess
from pathlib import Path
from typing import Any

import torch

# == provenance ================================================================


def commit_hash(short: bool = True) -> str:

    command = (
        ["git", "rev-parse", "--short", "HEAD"]
        if short
        else ["git", "rev-parse", "HEAD"]
    )

    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


# == memory ====================================================================


def gpu_used_mb(device: str) -> float:
    """Whole-process device usage, not just torch's.

    XGBoost allocates through its own CUDA allocator, so `max_memory_allocated`
    cannot see the ellpack or the histograms -- the numbers that matter when the
    question is whether a run fits. `mem_get_info` reads the driver, which sees
    every allocation on the card.
    """

    if not device.startswith("cuda"):
        return 0.0

    free, total = torch.cuda.mem_get_info()

    return (total - free) / 1e6


def gpu_total_mb(device: str) -> float:

    return torch.cuda.mem_get_info()[1] / 1e6 if device.startswith("cuda") else 0.0


def host_peak_rss_mb() -> float:

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def host_rss_mb() -> float:
    """Resident set *now*, unlike `host_peak_rss_mb`'s high-water mark.

    A streaming run's RSS rises with the file pages it has touched and falls
    when they are dropped, so the peak alone cannot show whether dropping is
    working.
    """

    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * resource.getpagesize() / 1e6
    except (OSError, IndexError, ValueError):
        return 0.0


def _meminfo(field: str) -> float:

    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(field):
                    return float(line.split()[1]) / 1024
    except OSError:
        pass

    return 0.0


def host_free_mb() -> float:
    """MemFree -- the number watchdogs tend to read, unlike MemAvailable.

    Worth logging next to MemAvailable precisely because they disagree during a
    streaming run, and it is the pessimistic one that gets runs killed.
    """

    return _meminfo("MemFree:")


def host_available_mb() -> float:
    """MemAvailable, not MemFree.

    A streaming run holds most of its footprint as reclaimable page cache, so
    MemFree reads as almost nothing while the machine is not short of memory at
    all. MemAvailable is the number that says whether the next allocation will
    succeed.
    """

    return _meminfo("MemAvailable:")


# == curves ====================================================================


def curve(y: list[float], x: list[int] | None = None) -> dict[str, list]:
    """One metric over one split, as the result files store it."""

    return {
        "x": list(range(len(y))) if x is None else [int(v) for v in x],
        "y": [float(v) for v in y],
    }


# == writing ===================================================================


def write_results(out: Path, models: dict[str, Any], info: dict[str, Any]) -> Path:
    """Merge model entries into a dataset's results file.

    Read-modify-write rather than overwrite, so a later run of another model --
    or a rerun of this one -- accumulates into the same file instead of
    discarding what is already there. Callers write after every model for the
    same reason `experiment.py` does: these runs are long, and a crash in the
    expensive one should not cost the cheap ones.
    """

    payload: dict[str, Any] = {}

    if out.exists():
        payload = json.loads(out.read_text())

    payload.update({k: v for k, v in info.items() if k != "models"})
    payload.setdefault("models", {}).update(models)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=str))

    return out
