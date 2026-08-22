"""Compare the torch implementation against the numba reference.

uv run python -m fit2082.boost.benchmark --epochs 50
"""

import argparse
import time

import numpy as np
import torch

from fit2082.boost import HashBoost
from fit2082.demo.boost import NewHashBoost
from fit2082.demo.utils import Dataset

# == data ======================================================================


def load(path: str, dataset: str, num_train: int, num_valid: int, batch_size: int):

    data = Dataset(
        f"{path}/{dataset}/{dataset}_X.npy", f"{path}/{dataset}/{dataset}_y.npy"
    )
    data.batch_size = batch_size

    num_classes = len(data.classes)

    fold = np.loadtxt(f"{path}/{dataset}/test_indices_fold_0.txt")
    indices = np.setdiff1d(np.arange(data.shape[0]), fold)

    np.random.seed(123)
    np.random.shuffle(indices)

    training = data[indices[:num_train]]
    validation = data[indices[num_train : num_train + num_valid]]

    batches = [
        (X.reshape(X.shape[0], -1).astype(np.float32), Y.astype(np.int32))
        for X, Y in training
    ]

    for X, Y in validation:
        X_VA = X.reshape(X.shape[0], -1).astype(np.float32)
        Y_VA = Y.astype(np.int32)
        break

    data.close()

    return batches, X_VA, Y_VA, num_classes


# == runs ======================================================================


def run_numba(batches, X_VA, Y_VA, num_classes, epochs, num_bits, lr):

    model = NewHashBoost(
        num_classes=np.int32(num_classes),
        num_pairs_per_hash=np.int32(num_bits),
        lr=np.float32(lr),
        max_num_hashes=epochs * len(batches) + 1,
    )

    start = time.time()

    for _ in range(epochs):
        for X, Y in batches:
            model.fit_batch(X, Y)

    elapsed = time.time() - start
    error = (model.predict_proba(X_VA).argmax(-1) != Y_VA).mean()

    return elapsed, float(error), model.num_rounds


def run_torch(batches, X_VA, Y_VA, num_classes, epochs, num_bits, lr, device, compile):

    on_device = [
        (
            torch.as_tensor(X, device=device),
            torch.as_tensor(Y.astype(np.int64), device=device),
        )
        for X, Y in batches
    ]
    X_VA_d = torch.as_tensor(X_VA, device=device)

    model = HashBoost(
        num_classes=num_classes,
        num_bits=num_bits,
        lr=lr,
        max_num_hashes=epochs * len(batches) + 1,
        device=device,
        compile=compile,
    )

    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    start = time.time()

    for _ in range(epochs):
        for X, Y in on_device:
            model.fit_batch(X, Y)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.time() - start

    error = (model.predict_proba(X_VA_d).argmax(-1).cpu().numpy() != Y_VA).mean()

    peak = torch.cuda.max_memory_allocated() / 1e6 if device.startswith("cuda") else 0.0

    return elapsed, float(error), model.num_rounds, peak


# == main ======================================================================


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="data")
    parser.add_argument("--dataset", default="Pedestrian")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--num-train", type=int, default=32768 * 2)
    parser.add_argument("--num-valid", type=int, default=4096)
    parser.add_argument("--num-bits", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-numba", action="store_true")
    args = parser.parse_args()

    batches, X_VA, Y_VA, num_classes = load(
        args.path, args.dataset, args.num_train, args.num_valid, args.batch_size
    )

    print(
        f"{args.dataset}: {len(batches)} batches of {batches[0][0].shape}, "
        f"{num_classes} classes, {args.epochs} epochs "
        f"-> {args.epochs * len(batches)} rounds"
    )

    baseline = None

    if not args.skip_numba:
        elapsed, error, rounds = run_numba(
            batches, X_VA, Y_VA, num_classes, args.epochs, args.num_bits, args.lr
        )
        baseline = elapsed
        print(f"  numba (cpu)   {elapsed:7.2f}s  rounds={rounds}  val_err={error:.4f}")

    elapsed, error, rounds, peak = run_torch(
        batches,
        X_VA,
        Y_VA,
        num_classes,
        args.epochs,
        args.num_bits,
        args.lr,
        args.device,
        args.compile,
    )

    speedup = f"  speedup={baseline / elapsed:.1f}x" if baseline else ""
    memory = f"  peak={peak:.0f}MB" if peak else ""

    print(
        f"  torch ({args.device})  {elapsed:7.2f}s  rounds={rounds}  "
        f"val_err={error:.4f}{memory}{speedup}"
    )


if __name__ == "__main__":
    main()
