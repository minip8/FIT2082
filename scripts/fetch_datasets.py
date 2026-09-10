"""Fetch MONSTER datasets from Hugging Face into ``data/``.

The ``monster-monash`` repositories already store each dataset in the layout
this project expects, so fetching is just a matter of pulling the right files:

    data/<Name>/<Name>_X.npy
    data/<Name>/<Name>_y.npy
    data/<Name>/test_indices_fold_<k>.txt

which is the layout ``fit2082.boost.experiment.load_split`` reads. The CSV
mirrors of X and y in each repository are ignored -- they are several times
larger and nothing here reads them.

Usage:

    uv run scripts/fetch_datasets.py --list
    uv run scripts/fetch_datasets.py Traffic LenDB
    uv run scripts/fetch_datasets.py AudioMNIST --folds all
    uv run scripts/fetch_datasets.py Traffic --dry-run

`--stdin` reads the same arguments from a pipe, so a list of names kept in a
file does not have to be pasted onto a command line:

    uv run scripts/fetch_datasets.py --stdin --folds all < datasets.txt
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download, list_repo_files
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError

from fit2082.cli import parse_args

# == datasets ==================================================================

OWNER = "monster-monash"

# The 29 datasets in MONSTER (https://arxiv.org/abs/2502.15122), by category.
DATASETS = {
    "AudioMNIST": "audio",
    "AudioMNIST-DS": "audio",
    "CornellWhaleChallenge": "audio",
    "FruitFlies": "audio",
    "InsectSound": "audio",
    "MosquitoSound": "audio",
    "WhaleSounds": "audio",
    "CrowdSourced": "eeg",
    "DREAMERA": "eeg",
    "DREAMERV": "eeg",
    "STEW": "eeg",
    "Opportunity": "har",
    "PAMAP2": "har",
    "Skoda": "har",
    "UCIActivity": "har",
    "USCActivity": "har",
    "WISDM": "har",
    "WISDM2": "har",
    "LenDB": "other",
    "FordChallenge": "other",
    "Pedestrian": "count",
    "Traffic": "count",
    "LakeIce": "satellite",
    "S2Agri-10pc-17": "satellite",
    "S2Agri-10pc-34": "satellite",
    "S2Agri-17": "satellite",
    "S2Agri-34": "satellite",
    "TimeSen2Crop": "satellite",
    "Tiselac": "satellite",
}

NUM_FOLDS = 5


def resolve(name: str) -> str:
    """Match a dataset name case-insensitively, or exit with the near misses."""

    for known in DATASETS:
        if known.lower() == name.lower():
            return known

    close = [k for k in DATASETS if name.lower() in k.lower()]
    hint = f" Did you mean: {', '.join(close)}?" if close else ""
    sys.exit(f"unknown dataset {name!r}.{hint} Use --list to see all 29.")


# == fetching ==================================================================


def wanted(dataset: str, folds: list[int]) -> list[str]:

    return [
        f"{dataset}_X.npy",
        f"{dataset}_y.npy",
        *[f"test_indices_fold_{k}.txt" for k in folds],
    ]


def sizes(dataset: str, files: list[str]) -> dict[str, int]:
    """Look up the remote size of each file, keyed by filename."""

    info = HfApi().dataset_info(f"{OWNER}/{dataset}", files_metadata=True)
    sizes = {f.rfilename: (f.size or 0) for f in info.siblings or []}

    return {f: sizes.get(f, 0) for f in files}


def human(num_bytes: int) -> str:

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def fetch(dataset: str, out: Path, folds: list[int], force: bool) -> Path:

    repo = f"{OWNER}/{dataset}"
    target = out / dataset
    target.mkdir(parents=True, exist_ok=True)

    available = set(list_repo_files(repo, repo_type="dataset"))

    for name in wanted(dataset, folds):
        if name not in available:
            print(f"  ! {name} not in {repo}, skipping")
            continue

        if not force and (target / name).exists():
            print(f"  = {name} (already present)")
            continue

        print(f"  + {name}")
        hf_hub_download(
            repo,
            name,
            repo_type="dataset",
            local_dir=target,
        )

    # hf_hub_download leaves resume metadata behind under local_dir; drop it so
    # the directory holds exactly the files data/Pedestrian does.
    shutil.rmtree(target / ".cache", ignore_errors=True)

    return target


def summarise(dataset: str, target: Path) -> None:
    """Print the shapes actually on disk, as a check the fetch is usable."""

    path_X, path_y = target / f"{dataset}_X.npy", target / f"{dataset}_y.npy"
    if not (path_X.exists() and path_y.exists()):
        print("  ! X or y missing, dataset is not loadable")
        return

    X = np.load(path_X, mmap_mode="r")
    y = np.load(path_y, mmap_mode="r")

    print(
        f"  X {X.shape} {X.dtype}, y {y.shape} {y.dtype}, {len(np.unique(y))} classes"
    )

    if X.shape[0] != y.shape[0]:
        print(f"  ! X and y disagree on length: {X.shape[0]} vs {y.shape[0]}")


# == main ======================================================================


def main() -> None:

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", nargs="*", help="dataset names, e.g. Traffic")
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument(
        "--folds",
        default="0",
        help="fold indices to fetch: '0' (default), '0,2', or 'all'",
    )
    parser.add_argument("--all", action="store_true", help="fetch every dataset")
    parser.add_argument("--list", action="store_true", help="list datasets and exit")
    parser.add_argument("--force", action="store_true", help="re-download existing")
    parser.add_argument(
        "--dry-run", action="store_true", help="print download sizes and exit"
    )
    args = parse_args(parser)

    if args.list:
        width = max(len(name) for name in DATASETS)
        for name, category in sorted(DATASETS.items(), key=lambda kv: (kv[1], kv[0])):
            print(f"{name:<{width}}  {category}")
        return

    if args.all:
        names = list(DATASETS)
    elif args.datasets:
        names = [resolve(name) for name in args.datasets]
    else:
        parser.error("name at least one dataset, or pass --all or --list")

    if args.folds == "all":
        folds = list(range(NUM_FOLDS))
    else:
        folds = [int(k) for k in args.folds.replace(",", " ").split()]

    for name in names:
        print(f"{name}:")
        try:
            if args.dry_run:
                total = 0
                for filename, size in sizes(name, wanted(name, folds)).items():
                    print(f"  {human(size):>12}  {filename}")
                    total += size
                print(f"  {human(total):>12}  total")
                continue

            target = fetch(name, args.data_dir, folds, args.force)
            summarise(name, target)
        except RepositoryNotFoundError:
            print(f"  ! no repository {OWNER}/{name}")
        except EntryNotFoundError as error:
            print(f"  ! {error}")


if __name__ == "__main__":
    main()
