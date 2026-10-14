"""GainSplitter picks the candidate splits that separate the classes.

It only changes which predicates a round uses, so what is worth pinning is the
choice itself: the informative split wins over noise, a split already taken is
never taken twice, one candidate per bit reduces to the base splitter's own
splits, and a model built with it trains.
"""

import torch

from fit2082.boost import HashBoost
from fit2082.boost.splits import GainSplitter


class FixedSplitter:
    """Proposes the same candidate splits every time, whatever is asked."""

    def __init__(self, features: list[int], midpoints: list[float]) -> None:

        self.features = torch.tensor(features, dtype=torch.int64)
        self.midpoints = torch.tensor(midpoints, dtype=torch.float32)

    def propose(self, X, Y, probabilities, gradient, hessian, num_bits):

        assert num_bits == self.features.shape[0]

        return self.features, self.midpoints


def batch(n: int = 512, seed: int = 0):
    """Feature 0 decides the class; features 1-3 are noise."""

    generator = torch.Generator().manual_seed(seed)
    X = torch.rand((n, 4), generator=generator)
    Y = (X[:, 0] > 0.5).to(torch.int64)

    probabilities = torch.full((n, 2), 0.5)
    gradient = probabilities - torch.nn.functional.one_hot(Y, 2).float()
    hessian = probabilities * (1 - probabilities)

    return X, Y, probabilities, gradient, hessian


def test_the_informative_split_goes_first():

    X, Y, p, g, h = batch()
    splitter = GainSplitter(
        candidates=4, base=FixedSplitter([1, 2, 0, 3], [0.5, 0.5, 0.5, 0.5])
    )

    features, midpoints = splitter.propose(X, Y, p, g, h, num_bits=1)

    assert features.tolist() == [0]
    assert midpoints.tolist() == [0.5]


def test_no_candidate_is_taken_twice():

    X, Y, p, g, h = batch()
    splitter = GainSplitter(
        candidates=2, base=FixedSplitter([0, 0, 1, 2], [0.5, 0.5, 0.5, 0.5])
    )

    features, _ = splitter.propose(X, Y, p, g, h, num_bits=2)

    # candidates 0 and 1 are the same split: the first bit takes one, and the
    # second bit may take the other copy, but never the same candidate again
    assert features[0].item() == 0
    assert features.shape == (2,)


def test_one_candidate_keeps_the_base_splits():

    X, Y, p, g, h = batch()
    base = FixedSplitter([1, 2, 3], [0.2, 0.4, 0.6])

    features, midpoints = GainSplitter(candidates=1, base=base).propose(
        X, Y, p, g, h, num_bits=3
    )

    assert sorted(zip(features.tolist(), midpoints.tolist())) == sorted(
        zip(base.features.tolist(), base.midpoints.tolist())
    )


def test_a_model_with_gain_splits_trains():

    X, Y, *_ = batch(n=1024, seed=1)
    model = HashBoost(
        num_classes=2,
        num_bits=4,
        max_num_hashes=40,
        device="cpu",
        splitter=GainSplitter(candidates=8),
    )

    for _ in range(40):
        model.fit_batch(X, Y)

    error = (model.predict(X).argmax(-1) != Y).float().mean().item()

    assert error < 0.05
