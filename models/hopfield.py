"""Hopfield associative memories for continuous sensor data.

Two networks are implemented, and both are tested in this project.

1. ContinuousHopfield - the main model. A modern (dense) Hopfield network whose
   states are real numbers, so sensor readings are stored without rounding.
       Energy : E(x) = -(1/beta) log sum_i exp(-beta/2 * d_i(x))
       Update : x <- M' softmax(-beta/2 * d(x))
       where d_i(x) = sum_j w_j (x_j - m_ij)^2 is the squared distance from the
       state to memory i. Known neurons have w_j = 1, unknown ones w_j = gamma.
       With gamma = 1 this equals the energy of Ramsauer et al. (2020),
       1/2 x'x cancelled, with a bias b_i = -1/2 m_i'm_i on each memory.

2. BinaryHopfield - the classical 1982 network. Readings are first turned into
   +1/-1 bits with a thermometer code.
       Weights: W = (1/N) P'P (Hebbian)  or  W = P' (P P')^+ P (pseudo-inverse)
       Energy : E(s) = -1/2 s'W s
       Update : s_i <- sign(sum_j W_ij s_j), one neuron at a time

In both, a known reading is clamped (never changed) and only the unknown
neurons are updated, so the network completes a pattern from a partial cue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import KMeans


def build_windows(z: np.ndarray, context: int) -> np.ndarray:
    """Stack each hour with its `context` neighbours on both sides.

    Row t becomes [x(t-c) ... x(t) ... x(t+c)]. Positions beyond the ends of the
    series are NaN. With context=0 the input is returned unchanged.
    """
    if context == 0:
        return z
    n, d = z.shape
    padded = np.full((n + 2 * context, d), np.nan)
    padded[context : context + n] = z
    return np.hstack([padded[i : i + n] for i in range(2 * context + 1)])


def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def _logsumexp_rows(logits: np.ndarray) -> np.ndarray:
    top = logits.max(axis=1, keepdims=True)
    return top[:, 0] + np.log(np.exp(logits - top).sum(axis=1))


@dataclass
class RetrievalTrace:
    """What happened during retrieval, for plots and for proving convergence."""

    energy: np.ndarray  # mean energy per iteration
    change: np.ndarray  # mean size of the state update per iteration
    iterations: int


class ContinuousHopfield:
    """Modern Hopfield network that completes real-valued sensor patterns.

    Parameters
    ----------
    beta : inverse temperature. Large beta retrieves one sharp memory; small
        beta blends many memories.
    context : neighbouring hours included in each stored pattern. 0 means each
        pattern is a single hour; 1 means previous, current and next hour.
    context_weight : how strongly neighbouring hours count when comparing a
        query with a memory, relative to the current hour.
    free_weight : gamma, how strongly unknown neurons count in that comparison.
        Their values are only estimates, so they should count for less than
        real readings. gamma=1 treats them as equally trustworthy.
    max_iter : update steps after the first partial-cue recall.
    """

    def __init__(self, beta=4.0, context=0, context_weight=1.0, free_weight=0.05, max_iter=20, tol=1e-4):
        self.beta = float(beta)
        self.context = int(context)
        self.context_weight = float(context_weight)
        self.free_weight = float(free_weight)
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.memories: np.ndarray | None = None
        self.n_features = 0

    # ---- storage ---------------------------------------------------------- #
    def fit(self, z_train: np.ndarray) -> "ContinuousHopfield":
        """Store every complete training pattern as one memory.

        Patterns with natural gaps are skipped rather than filled, so the memory
        contains real observations only. Storing needs no gradient training: the
        memory matrix M is the data.
        """
        self.n_features = z_train.shape[1]
        windows = build_windows(z_train, self.context)
        memories = windows[~np.isnan(windows).any(axis=1)]
        if len(memories) == 0:
            raise ValueError("No complete patterns are available to store.")
        self._scale = self._feature_scale()
        self.memories = memories * self._scale
        self._memories_squared = (self.memories**2).T
        return self

    def _feature_scale(self) -> np.ndarray:
        """Per-position scale; neighbouring hours get sqrt(context_weight)."""
        width = 2 * self.context + 1
        scale = np.full((width, self.n_features), np.sqrt(self.context_weight))
        scale[self.context] = 1.0
        return scale.ravel()

    # ---- energy ----------------------------------------------------------- #
    def _logits(self, state: np.ndarray, weight: np.ndarray) -> np.ndarray:
        """-beta/2 times the weighted squared distance to every memory."""
        distance = (
            (weight * state**2).sum(axis=1, keepdims=True)
            - 2.0 * (weight * state) @ self.memories.T
            + weight @ self._memories_squared
        )
        return -0.5 * self.beta * distance

    def energy(self, state: np.ndarray, known: np.ndarray | None = None) -> np.ndarray:
        """Energy of each state (scaled space). Lower means closer to a memory."""
        weight = self._neuron_weight(state, known)
        return -_logsumexp_rows(self._logits(state, weight)) / self.beta

    def _neuron_weight(self, state, known) -> np.ndarray:
        """Known neurons count fully; unknown ones count `free_weight`."""
        if known is None:
            return np.ones_like(state)
        return np.where(known, 1.0, self.free_weight)

    # ---- retrieval -------------------------------------------------------- #
    def _partial_cue(self, state: np.ndarray, known: np.ndarray) -> np.ndarray:
        """First update: compare memories with the known neurons only.

        This is the update rule with unknown neurons given zero weight, so the
        placeholder values in those neurons cannot bias the comparison.
        """
        weights = _softmax_rows(self._logits(state, known.astype(float)))
        return np.where(known, state, weights @ self.memories)

    def retrieve(self, state, known, max_iter=None, batch_size=256, on_batch=None):
        """Run the update rule, clamping known neurons, until the state settles."""
        max_iter = self.max_iter if max_iter is None else max_iter
        out = np.empty_like(state)
        energy_rows, change_rows, longest = [], [], 0
        for start in range(0, len(state), batch_size):
            part = slice(start, start + batch_size)
            keep = known[part]
            weight = self._neuron_weight(state[part], keep)
            x = self._partial_cue(state[part], keep)
            logits = self._logits(x, weight)
            energies, changes = [-_logsumexp_rows(logits) / self.beta], []
            for _ in range(max_iter):
                updated = np.where(keep, x, _softmax_rows(logits) @ self.memories)
                step = np.abs(updated - x).sum(axis=1) / np.maximum((~keep).sum(axis=1), 1)
                x = updated
                logits = self._logits(x, weight)
                energies.append(-_logsumexp_rows(logits) / self.beta)
                changes.append(step)
                if step.max() < self.tol:
                    break
            out[part] = x
            longest = max(longest, len(changes))
            energy_rows.append(energies)
            change_rows.append(changes)
            if on_batch:
                on_batch(min(start + batch_size, len(state)) / len(state))
        return out, _merge_trace(energy_rows, change_rows, longest, len(state))

    def reconstruct(self, z: np.ndarray, unknown: np.ndarray, max_iter=None, on_batch=None):
        """Fill the unknown cells of a time-ordered table.

        z       : z-scored table; NaN wherever no reading exists
        unknown : True where the network must supply the value
        Returns (completed table, RetrievalTrace). Known cells are never altered.
        """
        hidden = np.where(unknown, np.nan, z)
        windows = build_windows(hidden, self.context)
        known = ~np.isnan(windows)
        state = np.where(known, windows, 0.0) * self._scale
        final, trace = self.retrieve(state, known, max_iter, on_batch=on_batch)
        centre = slice(self.context * self.n_features, (self.context + 1) * self.n_features)
        recalled = final[:, centre]
        return np.where(unknown | np.isnan(z), recalled, z), trace


def _merge_trace(energy_rows, change_rows, longest, total) -> RetrievalTrace:
    """Average per-batch curves; a finished batch stays at its final value."""
    energy = np.zeros(longest + 1)
    change = np.zeros(longest)
    for batch_energy, batch_change in zip(energy_rows, change_rows):
        for i in range(longest + 1):
            energy[i] += batch_energy[min(i, len(batch_energy) - 1)].sum()
        for i in range(min(longest, len(batch_change))):
            change[i] += batch_change[i].sum()
    return RetrievalTrace(energy / total, change / total, longest)


class ThermometerEncoder:
    """Turn a real value into +1/-1 bits, keeping "how far apart" meaningful.

    With levels=5, level 0 is ----, level 1 is +---, level 2 is ++-- and so on.
    Two values that are close differ in few bits, so Hamming distance (what a
    binary Hopfield network sees) tracks the real distance. An ordinary binary
    number would not do this: 7 (0111) and 8 (1000) differ in every bit.
    """

    def __init__(self, levels=16):
        self.levels = int(levels)

    def fit(self, z: np.ndarray) -> "ThermometerEncoder":
        quantiles = np.linspace(0, 1, self.levels + 1)[1:-1]
        self.edges = np.nanquantile(z, quantiles, axis=0).T  # (features, levels-1)
        self.centres = np.empty((z.shape[1], self.levels))
        for j in range(z.shape[1]):
            column = z[:, j][~np.isnan(z[:, j])]
            bins = np.searchsorted(self.edges[j], column, side="right")
            overall = np.median(column)
            self.centres[j] = [
                np.median(column[bins == b]) if (bins == b).any() else overall
                for b in range(self.levels)
            ]
        self.bits = self.levels - 1
        return self

    def encode(self, z: np.ndarray) -> np.ndarray:
        """(rows, features) -> (rows, features * bits) of +1/-1."""
        filled = np.nan_to_num(z, nan=0.0)
        spikes = filled[:, :, None] >= self.edges[None, :, :]
        return np.where(spikes, 1.0, -1.0).reshape(len(z), -1)

    def decode(self, s: np.ndarray) -> np.ndarray:
        """Count the +1 bits per sensor, then look up that level's typical value."""
        count = (s.reshape(len(s), -1, self.bits) > 0).sum(axis=2)
        return self.centres[np.arange(self.centres.shape[0])[None, :], count]


class BinaryHopfield:
    """Classical Hopfield network on +1/-1 states."""

    def __init__(self, rule="pseudoinverse"):
        if rule not in ("hebbian", "pseudoinverse"):
            raise ValueError(f"Unknown learning rule: {rule}")
        self.rule = rule
        self.weights: np.ndarray | None = None

    def fit(self, patterns: np.ndarray) -> "BinaryHopfield":
        """Compute the weight matrix W from patterns of shape (P, N)."""
        n = patterns.shape[1]
        if self.rule == "hebbian":
            w = patterns.T @ patterns / n
        else:
            # Projection onto the span of the patterns. Unlike the Hebbian rule
            # it copes with correlated patterns, which sensor data always has.
            w = patterns.T @ np.linalg.pinv(patterns @ patterns.T) @ patterns
        np.fill_diagonal(w, 0.0)  # no self-connections
        self.weights = w
        return self

    def energy(self, s: np.ndarray) -> np.ndarray:
        return -0.5 * np.einsum("ri,ij,rj->r", s, self.weights, s)

    def retrieve(self, s, free, max_sweeps=10, rng=None):
        """Asynchronous updates, one neuron at a time, over the free neurons.

        A symmetric W with a zero diagonal guarantees the energy never rises,
        so the network always settles. Rows are processed side by side, but each
        row's neurons are still updated strictly one after another.
        """
        rng = rng or np.random.default_rng(0)
        s = s.copy()
        energies, changes = [self.energy(s).mean()], []
        for _ in range(max_sweeps):
            flips = 0
            for i in rng.permutation(s.shape[1]):
                rows = free[:, i]
                if not rows.any():
                    continue
                field = s[rows] @ self.weights[:, i]
                new = np.where(field > 0, 1.0, np.where(field < 0, -1.0, s[rows, i]))
                flips += int((new != s[rows, i]).sum())
                s[rows, i] = new
            energies.append(self.energy(s).mean())
            changes.append(flips / max(free.sum(), 1))
            if flips == 0:
                break
        return s, RetrievalTrace(np.array(energies), np.array(changes), len(changes))


class BinaryHopfieldImputer:
    """Thermometer code + classical network, for completing sensor tables.

    A classical network holds only about 0.14 N Hebbian patterns (fewer when
    they are correlated), so thousands of hours cannot be stored one by one.
    K-means first reduces the training data to `n_prototypes` typical states and
    those are what the network stores.
    """

    def __init__(self, levels=16, n_prototypes=64, rule="pseudoinverse", max_sweeps=10, seed=0):
        self.levels = levels
        self.n_prototypes = n_prototypes
        self.rule = rule
        self.max_sweeps = max_sweeps
        self.seed = seed

    def fit(self, z_train: np.ndarray) -> "BinaryHopfieldImputer":
        complete = z_train[~np.isnan(z_train).any(axis=1)]
        if len(complete) == 0:
            raise ValueError("No complete patterns are available to store.")
        self.encoder = ThermometerEncoder(self.levels).fit(z_train)
        k = min(self.n_prototypes, len(complete))
        kmeans = KMeans(n_clusters=k, n_init=3, random_state=self.seed).fit(complete)
        self.prototypes = kmeans.cluster_centers_
        self.network = BinaryHopfield(self.rule).fit(self.encoder.encode(self.prototypes))
        self.start_value = np.nanmedian(z_train, axis=0)
        self.n_features = z_train.shape[1]
        return self

    def reconstruct(self, z: np.ndarray, unknown: np.ndarray):
        """Same interface as ContinuousHopfield.reconstruct."""
        missing = unknown | np.isnan(z)
        start = np.where(missing, self.start_value, z)
        state = self.encoder.encode(start)
        free = np.repeat(missing, self.encoder.bits, axis=1)
        final, trace = self.network.retrieve(
            state, free, self.max_sweeps, np.random.default_rng(self.seed)
        )
        return np.where(missing, self.encoder.decode(final), z), trace
