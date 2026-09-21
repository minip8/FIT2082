# Nestor Cabello, Lars Kulik
# PULSAR: Advancing Interval-Based Time Series Classification to
# State-of-the-Art Performance
# ICDM 2025
#
# A torch port of the feature transform in https://github.com/stevcabello/PULSAR
# (GPL-3.0), rewritten to run batched on the GPU as a front end for
# `fit2082.boost`, the way `fit2082.quant` is for QUANT. The classifier head
# (calibrated ExtraTrees + ridge) is dropped.
"""PULSAR features: pooled local statistics over dilated sliding intervals.

For each representation of the series (original, periodogram, first
difference, Burg AR coefficients) and each (length, dilation) interval, every
sliding segment is summarised by 7 local statistics. Each resulting statistic
sequence is split into 1, 2, 4 and 8 contiguous partitions and pooled by 9
operators. Pooling over the whole sequence (level 0) gives the *global*
features, which are always kept. Finer levels use a random 6 of the 9 operators
per partition, chosen once at fit time. Together with the raw representation
they form the *local* features, of which the top 40% by Fisher score are kept.

Departures from upstream, all deliberate:

* Multivariate input `(n, channels, length)` is handled channel by channel and
  the features concatenated, as QUANT does. Upstream is univariate.
* `fit` takes an iterable of labelled batches and accumulates the class
  statistics that the Fisher score and the scaler need, so selection sees the
  whole training split without ever holding its features at once.
* Moments are computed centred in float32 rather than as E[x^2] - E[x]^2 in
  float64. The thresholds (stdev 0 below var 1e-14 / 1e-6) are upstream's.
* The AR order is clamped to `length - 2`, where statsmodels would raise, and
  non-finite AR coefficients (not only NaN) are zeroed.
"""

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch

# == statistics ================================================================

# Upstream approximates medians and IQRs from a 64-bin histogram of each row --
# a CPU speed trick, but it defines the features, so it is kept.
HIST_BINS = 64


def hist_quantiles(
    Y: torch.Tensor, qs: tuple[float, ...], bins: int = HIST_BINS
) -> list[torch.Tensor]:
    """Histogram-approximated quantiles over the last dim.

    Upstream bins each row into `bins` equal-width bins between its min and max
    and returns the midpoint of the first bin whose cumulative count reaches
    `q * n`. That bin is the `ceil(q * n)`-th smallest bin index, so sorting
    the bin indices gives it without building the histogram.
    """

    n = Y.shape[-1]
    lo = Y.amin(-1, keepdim=True)
    width = (Y.amax(-1, keepdim=True) - lo) / bins
    flat = width == 0

    index = ((Y - lo) / torch.where(flat, 1.0, width)).floor().clamp(max=bins - 1)
    index = index.sort(-1).values

    out = []
    for q in qs:
        k = max(math.ceil(q * n), 1) - 1
        value = lo + (index[..., k : k + 1] + 0.5) * width
        out.append(torch.where(flat, lo, value).squeeze(-1))

    return out


def _std(Y: torch.Tensor, mean: torch.Tensor, floor: float) -> torch.Tensor:

    var = (Y - mean.unsqueeze(-1)).square().mean(-1)

    return torch.where(var > floor, var.clamp(min=0).sqrt(), 0.0)


def _slope(Y: torch.Tensor) -> torch.Tensor:
    """Least-squares slope against 0..n-1; 0 for a single point."""

    n = Y.shape[-1]
    if n < 2:
        return torch.zeros(Y.shape[:-1], device=Y.device, dtype=Y.dtype)

    t = torch.arange(n, device=Y.device, dtype=Y.dtype) - (n - 1) / 2

    return (Y * t).sum(-1) / t.square().sum()


def local_stats(S: torch.Tensor) -> torch.Tensor:
    """(..., segments, length) -> (..., 7, segments).

    Order: mean, stdev, slope, min, max, median, iqr.
    """

    mean = S.mean(-1)
    q1, median, q3 = hist_quantiles(S, (0.25, 0.5, 0.75))

    stats = (
        mean,
        _std(S, mean, 1e-14),
        _slope(S),
        S.amin(-1),
        S.amax(-1),
        median,
        q3 - q1,
    )

    return torch.stack(stats, -2)


NUM_LOCAL_STATS = 7

# == pooling ===================================================================

POOLING_OPERATORS = (
    "max",
    "mean",
    "min",
    "median",
    "iqr",
    "stdev",
    "mean_crossing",
    "above_mean",
    "slope",
)


def pool(Y: torch.Tensor) -> torch.Tensor:
    """All 9 pooling operators over the last dim: (..., n) -> (..., 9)."""

    n = Y.shape[-1]
    lo, hi = Y.amin(-1), Y.amax(-1)
    # a float32 mean of a constant row can land a rounding error below it,
    # which would count every value as "above the mean"; upstream sums in
    # float64 and gets the constant back exactly
    mean = torch.minimum(torch.maximum(Y.mean(-1), lo), hi)
    m = mean.unsqueeze(-1)
    q1, median, q3 = hist_quantiles(Y, (0.25, 0.5, 0.75))

    if n > 1:
        prev, cur = Y[..., :-1], Y[..., 1:]
        crossings = ((prev <= m) & (cur > m)) | ((prev >= m) & (cur < m))
        crossing = crossings.sum(-1) / (n - 1)
    else:
        crossing = torch.zeros_like(mean)

    pooled = (
        hi,
        mean,
        lo,
        median,
        q3 - q1,
        _std(Y, mean, 1e-6),
        crossing.to(Y.dtype),
        (Y > m).to(Y.dtype).mean(-1),
        _slope(Y),
    )

    return torch.stack(pooled, -1)


def pool_partitions(T: torch.Tensor, parts: int) -> torch.Tensor:
    """Pool `parts` near-equal contiguous partitions: (..., n) -> (..., parts, 9).

    As in upstream's `get_partitions`, the first `n % parts` partitions are one
    longer than the rest. Partitions of equal width are pooled in one call.
    """

    s, r = divmod(T.shape[-1], parts)
    blocks = []

    if r:
        blocks.append(pool(T[..., : r * (s + 1)].unflatten(-1, (r, s + 1))))
    if parts > r:
        blocks.append(pool(T[..., r * (s + 1) :].unflatten(-1, (parts - r, s))))

    return torch.cat(blocks, -2)


# == representations ===========================================================

REPRESENTATIONS = ("original", "periodogram", "derivative", "autoregressive")


def ar_order(length: int) -> int:
    """Upstream's Burg order, clamped to what the recursion supports."""

    return min(int(12 * (length / 100) ** 0.25), length - 2)


def burg(X: torch.Tensor, order: int) -> torch.Tensor:
    """AR coefficients by Burg's method, batched over rows: (R, n) -> (R, order).

    A line-for-line vectorisation of statsmodels' `pacf_burg` followed by
    `levinson_durbin_pacf`, which is what upstream's per-row `burg` call runs.
    Computed in float64: the recursion divides by shrinking residual energies.
    """

    x = X.double()
    x = x - x.mean(-1, keepdim=True)
    R = x.shape[0]

    d = x.new_zeros(R, order + 1)
    pacf = x.new_zeros(R, order + 1)

    u = x.flip(-1)
    v = u.clone()

    d[:, 0] = 2 * x.square().sum(-1)
    d[:, 1] = u[:, :-1].square().sum(-1) + v[:, 1:].square().sum(-1)
    pacf[:, 1] = 2 / d[:, 1] * (v[:, 1:] * u[:, :-1]).sum(-1)

    for i in range(1, order):
        k = pacf[:, i : i + 1]
        u, v = (
            torch.cat([u[:, :1], u[:, :-1] - k * v[:, 1:]], -1),
            torch.cat([v[:, :1], v[:, 1:] - k * u[:, :-1]], -1),
        )
        d[:, i + 1] = (1 - pacf[:, i] ** 2) * d[:, i] - v[:, i] ** 2 - u[:, -1] ** 2
        pacf[:, i + 1] = 2 / d[:, i + 1] * (v[:, i + 1 :] * u[:, i:-1]).sum(-1)

    ar = pacf[:, 1:].clone()
    for i in range(1, order):
        prev = ar[:, :i].clone()
        ar[:, :i] = prev - ar[:, i : i + 1] * prev.flip(-1)

    return torch.nan_to_num(ar, nan=0.0, posinf=0.0, neginf=0.0).to(X.dtype)


def represent(X: torch.Tensor, name: str) -> torch.Tensor:
    """(R, n) -> (R, n') for one representation."""

    n = X.shape[-1]

    if name == "original":
        return X
    if name == "periodogram":
        return torch.fft.rfft(X).abs()[..., : n // 2]
    if name == "derivative":
        return X.diff()
    if name == "autoregressive":
        order = ar_order(n)
        return burg(X, order) if order >= 1 else X[..., :0]

    raise ValueError(f"unknown representation {name!r}")


# == fitted structure ==========================================================


@dataclass
class Interval:
    """One (length, dilation) interval and its fit-time pooling choices."""

    length: int
    dilation: int
    num_segments: int
    levels: int
    # flat indices into this interval's (7, partitions, 9) local pool, over
    # levels 1.. : operators drawn for each partition, single-point ones dropped
    local_index: torch.Tensor

    @property
    def has_global(self) -> bool:
        return self.num_segments > 1

    def segments(self, Z: torch.Tensor) -> torch.Tensor:
        """(R, n) -> (R, segments, length), every sliding dilated window."""

        span = (self.length - 1) * self.dilation + 1

        return Z.unfold(-1, span, 1)[..., :: self.dilation]


def make_intervals(
    input_length: int,
    lengths: tuple[int, ...],
    max_dilation: int | None,
    depth: int,
    num_ops: int,
    generator: torch.Generator,
) -> list[Interval]:
    """Upstream's `generate_fixed_intervals` plus its per-partition operator draws."""

    intervals = []

    for length in lengths:
        if length > input_length:
            continue

        max_exponent = math.log2(max((input_length - 1) // (length - 1), 1))
        if max_dilation:
            max_exponent = min(max_exponent, math.log2(max_dilation))

        for e in range(math.floor(max_exponent) + 1):
            dilation = 2**e
            num_segments = input_length - (length - 1) * dilation
            levels = min(int(math.log2(num_segments)) + 1, depth)

            masks = []
            for level in range(1, levels):
                parts = 2**level
                s, r = divmod(num_segments, parts)
                for p in range(parts):
                    chosen = torch.zeros(len(POOLING_OPERATORS), dtype=torch.bool)
                    chosen[
                        torch.randperm(len(POOLING_OPERATORS), generator=generator)[
                            :num_ops
                        ]
                    ] = True
                    width = s + 1 if p < r else s
                    masks.append(chosen & (width > 1))

            if masks:
                mask = torch.stack(masks).expand(NUM_LOCAL_STATS, -1, -1)
                local_index = mask.flatten().nonzero().squeeze(-1)
            else:
                local_index = torch.zeros(0, dtype=torch.long)

            intervals.append(
                Interval(length, dilation, num_segments, levels, local_index)
            )

    return intervals


# == pulsar ====================================================================


class Pulsar:
    """Supervised PULSAR feature transform over `(n, channels, length)` tensors.

    Defaults are upstream's. `fit` sees every training batch once;
    `transform` then gives `[global, selected local]`, standardised.
    """

    def __init__(
        self,
        lengths: tuple[int, ...] = (7, 9, 11),
        depth: int = 4,
        top_percent: float = 40,
        num_ops: int = 6,
        max_dilation: int | None = 16,
        standardise: bool = True,
        seed: int = 0,
        chunk_elements: int = 1 << 25,
    ) -> None:

        assert all(length >= 2 for length in lengths)
        assert depth >= 1
        assert 0 < top_percent <= 100

        self.lengths = tuple(lengths)
        self.depth = depth
        self.top_percent = top_percent
        self.num_ops = num_ops
        self.max_dilation = max_dilation
        self.standardise = standardise
        self.seed = seed
        # rows per chunk are sized so the unselected feature matrix stays below
        # this many floats; the per-interval intermediates are of the same order
        self.chunk_elements = chunk_elements

        self.intervals: dict[str, list[Interval]] = {}
        self.fitted = False

    # -- structure -------------------------------------------------------------

    def _build(self, shape: torch.Size) -> None:

        _, self.channels, self.input_length = shape

        generator = torch.Generator().manual_seed(self.seed)
        probe = torch.zeros(1, self.input_length)

        self.intervals = {}
        self.num_global = 0
        self.num_local = 0

        for name in REPRESENTATIONS:
            n = represent(probe, name).shape[-1]
            intervals = make_intervals(
                n,
                self.lengths,
                self.max_dilation,
                self.depth,
                self.num_ops,
                generator,
            )
            self.intervals[name] = intervals

            for interval in intervals:
                if interval.has_global:
                    self.num_global += NUM_LOCAL_STATS * len(POOLING_OPERATORS)
                self.num_local += len(interval.local_index)
            self.num_local += n

    # -- features --------------------------------------------------------------

    def _unselected(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(n, c, L) -> global (n, c * G), all local (n, c * L_all)."""

        num, channels, length = X.shape
        assert (channels, length) == (self.channels, self.input_length), (
            f"fitted on {(self.channels, self.input_length)}, got {(channels, length)}"
        )

        rows = X.reshape(num * channels, length).float()
        global_parts = []
        local_parts = []

        for name, intervals in self.intervals.items():
            Z = represent(rows, name)

            for interval in intervals:
                T = local_stats(interval.segments(Z))

                if interval.has_global:
                    global_parts.append(pool_partitions(T, 1).flatten(1))

                if interval.levels > 1:
                    pooled = torch.cat(
                        [
                            pool_partitions(T, 2**level)
                            for level in range(1, interval.levels)
                        ],
                        -2,
                    )
                    index = interval.local_index.to(pooled.device)
                    local_parts.append(pooled.flatten(1)[:, index])

            local_parts.append(Z)

        empty = rows[:, :0]
        G = torch.cat(global_parts, -1) if global_parts else empty
        L = torch.cat(local_parts, -1) if local_parts else empty

        return G.reshape(num, -1), L.reshape(num, -1)

    def _chunk_rows(self) -> int:

        width = max(self.channels * (self.num_global + self.num_local), 1)

        return max(self.chunk_elements // width, 1)

    # -- fit -------------------------------------------------------------------

    def fit(self, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> "Pulsar":
        """Accumulate per-class moments of every feature, then select and scale.

        Moments are float64 sums of values shifted by the first chunk's column
        means, which keeps the variance subtraction well conditioned.
        """

        self.intervals = {}
        moments: ClassMoments | None = None

        for X, Y in batches:
            if not self.intervals:
                self._build(X.shape)

            step = self._chunk_rows()
            for Xc, Yc in zip(X.split(step), Y.split(step), strict=True):
                F = torch.cat(self._unselected(Xc), -1)
                if moments is None:
                    moments = ClassMoments(F)
                moments.add(F, Yc.long())

        assert moments is not None, "fit needs at least one batch"
        count, total, squares = moments.count, moments.total, moments.squares
        shift = moments.shift

        num_global = self.channels * self.num_global
        scores = fisher_scores(count, total[:, num_global:], squares[:, num_global:])

        keep = int(self.top_percent / 100 * scores.numel())
        self.selected = (
            scores.argsort(descending=True, stable=True)[:keep].sort().values
        )

        columns = torch.cat(
            [
                torch.arange(num_global, device=scores.device),
                num_global + self.selected,
            ]
        )

        N = count.sum()
        mean = total[:, columns].sum(0) / N
        var = (squares[:, columns].sum(0) / N - mean.square()).clamp(min=0)
        scale = var.sqrt()
        # sklearn's StandardScaler leaves (near-)constant columns unscaled
        scale = torch.where(scale < 10 * torch.finfo(scale.dtype).eps, 1.0, scale)

        self.mean = (shift[columns] + mean).float()
        self.scale = scale.float()
        self.fitted = True

        return self

    # -- transform -------------------------------------------------------------

    @property
    def num_features(self) -> int:

        assert self.fitted, "not fitted"

        return self.channels * self.num_global + len(self.selected)

    def transform(self, X: torch.Tensor) -> torch.Tensor:

        assert self.fitted, "not fitted"

        out = []
        for Xc in X.split(self._chunk_rows()):
            G, L = self._unselected(Xc)
            Z = torch.cat([G, L[:, self.selected.to(L.device)]], -1)
            if self.standardise:
                Z = (Z - self.mean.to(Z.device)) / self.scale.to(Z.device)
            out.append(Z)

        return torch.cat(out)

    def fit_transform(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:

        return self.fit([(X, Y)]).transform(X)


# == feature selection =========================================================


class ClassMoments:
    """Per-class count, sum and sum of squares of every column, in float64.

    Values are shifted by the first chunk's column means before summing, which
    keeps the later variance subtraction well conditioned. Classes are added as
    their labels first appear.
    """

    def __init__(self, first: torch.Tensor) -> None:

        self.shift = first.double().mean(0)
        self.count = self.shift.new_zeros(0)
        self.total = self.shift.new_zeros(0, len(self.shift))
        self.squares = self.shift.new_zeros(0, len(self.shift))

    def add(self, F: torch.Tensor, Y: torch.Tensor) -> None:

        grow = int(Y.max()) + 1 - len(self.count)
        if grow > 0:
            self.count = torch.cat([self.count, self.count.new_zeros(grow)])
            self.total = torch.cat(
                [self.total, self.total.new_zeros(grow, len(self.shift))]
            )
            self.squares = torch.cat(
                [self.squares, self.squares.new_zeros(grow, len(self.shift))]
            )

        F = F.double() - self.shift
        self.count.index_add_(0, Y, torch.ones_like(Y, dtype=F.dtype))
        self.total.index_add_(0, Y, F)
        self.squares.index_add_(0, Y, F.square())


def fisher_scores(
    count: torch.Tensor, total: torch.Tensor, squares: torch.Tensor
) -> torch.Tensor:
    """Upstream's Fisher score per column, from per-class moment sums.

    `count` is (classes,), `total` and `squares` are (classes, columns) sums of
    (shifted) values and their squares. Classes never seen are ignored. The
    per-class sd is the sample sd (ddof=1), 1e-5 for a single example, floored
    at 1e-4; a column scores 0 when its numerator or denominator is 0.
    """

    present = count > 0
    n = count[present].unsqueeze(-1)
    total = total[present]
    squares = squares[present]

    class_mean = total / n
    mean = total.sum(0) / n.sum()

    var = (squares - n * class_mean.square()) / (n - 1).clamp(min=1)
    sd = torch.where(n > 1, var.clamp(min=0).sqrt(), 1e-5).clamp(min=1e-4)

    numerator = (n * (class_mean - mean).square()).sum(0)
    denominator = (n * sd.square()).sum(0)

    return torch.where(
        (numerator == 0) | (denominator == 0), 0.0, numerator / denominator
    )
