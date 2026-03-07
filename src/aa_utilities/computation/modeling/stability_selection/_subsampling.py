# subsampling.py
from __future__ import annotations
from typing import Optional, Sequence, Tuple
import numpy as np


class BaseSampler:
    """
    Strategy interface for subsampling.
    Each sampler implements draw(n, rng, labels=None) and returns a tuple of index arrays.
    - For single-fit strategies (standard/stratified/bootstrap): returns a 1-tuple (idx,).
    - For CPSS: returns a 2-tuple (idx_a, idx_b) representing two complementary halves.
    """

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError("Subclasses must implement draw().")


class StandardSampler(BaseSampler):
    """
    Uniform subsampling without replacement.
    - ratio: fraction of samples to draw (0 < ratio <= 1).
    """

    def __init__(self, ratio: float = 0.5):
        if not (0 < ratio <= 1.0):
            raise ValueError("ratio must be in (0, 1].")
        self.ratio = ratio

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, ...]:
        k = max(1, int(round(self.ratio * n)))
        idx = rng.choice(n, size=k, replace=False)
        return (np.sort(idx),)


class BootstrapSampler(BaseSampler):
    """
    Ratio-based bootstrap subsampling.
    - replacement: whether to sample with replacement (duplicates possible).
    - ratio: fraction of samples to draw (0 < ratio <= 1).
    Notes:
    - If replacement=True, indices may repeat and are returned in draw order.
    - If replacement=False, behavior matches StandardSampler except the parameter name signals intent.
    """

    def __init__(self, replacement: bool = True, ratio: float = 0.5):
        if not (0 < ratio <= 1.0):
            raise ValueError("ratio must be in (0, 1].")
        self.replacement = replacement
        self.ratio = ratio

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, ...]:
        k = max(1, int(round(self.ratio * n)))
        idx = rng.choice(n, size=k, replace=self.replacement)
        # Keep order if replacement=True (preserves multiplicity sequencing). Otherwise, sort for determinism.
        return (idx if self.replacement else np.sort(idx),)


class StratifiedSampler(BaseSampler):
    """
    Stratified subsampling by group labels with strict proportional enforcement.
    - groups: a length-n 1D array aligned to samples; fixed per run.
    - ratio: fraction of samples to draw (0 < ratio <= 1).
    Behavior:
    - Compute per-group targets: k_g = round(ratio * n_g).
    - Adjust targets so sum(k_g) == round(ratio * n) via largest-remainder correction.
    - Draw within groups without replacement using the provided RNG.
    - Strict proportional enforcement: groups for which k_g == 0 contribute no samples.
    - Raises an error if any k_g > n_g (should not occur unless ratio misconfigured).
    """

    def __init__(self, groups: Sequence[str], ratio: float = 0.5):
        if not (0 < ratio <= 1.0):
            raise ValueError("ratio must be in (0, 1].")
        groups = np.asarray(groups)
        if groups.ndim != 1:
            raise ValueError("groups must be a 1D array-like aligned to samples.")
        if len(groups) == 0:
            raise ValueError("groups cannot be empty.")
        
        # Validate that ratio is reasonable for stratification
        unique_groups, counts = np.unique(groups, return_counts=True)
        min_group_size = counts.min()
        
        # Warn if ratio is too small for meaningful stratification
        if ratio * min_group_size < 0.5:
            import warnings
            warnings.warn(
                f"StratifiedSampler: ratio={ratio} is too small for the smallest group "
                f"(size={min_group_size}). Expected samples: {ratio * min_group_size:.2f}. "
                f"This may lead to groups being excluded from sampling, violating stratification. "
                f"Consider using a larger ratio (>= {1.0 / min_group_size:.3f}) to ensure all groups contribute.",
                UserWarning,
                stacklevel=2
            )
        
        self.groups = groups
        self.ratio = ratio

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, ...]:
        if n != self.groups.shape[0]:
            raise ValueError("StratifiedSampler: len(groups) must equal n samples in X.")

        unique_groups, inverse = np.unique(self.groups, return_inverse=True)
        # Map each group to the indices of its members
        group_indices = {g: np.where(inverse == i)[0] for i, g in enumerate(unique_groups)}
        group_sizes = {g: len(ix) for g, ix in group_indices.items()}

        total_k = max(1, int(round(self.ratio * n)))
        # Initial targets
        raw_targets = {g: self.ratio * group_sizes[g] for g in unique_groups}
        target_counts = {g: int(round(raw_targets[g])) for g in unique_groups}
        current_sum = sum(target_counts.values())

        # Largest-remainder correction to match sum(target_counts) == total_k
        if current_sum != total_k:
            remainders = {g: raw_targets[g] - round(raw_targets[g]) for g in unique_groups}
            if current_sum < total_k:
                # Add one to groups with largest positive remainders
                for g in sorted(unique_groups, key=lambda gg: remainders[gg], reverse=True):
                    if current_sum == total_k:
                        break
                    target_counts[g] += 1
                    current_sum += 1
            else:
                # Remove one from groups with smallest (most negative) remainders first
                for g in sorted(unique_groups, key=lambda gg: remainders[gg]):
                    if current_sum == total_k:
                        break
                    if target_counts[g] > 0:
                        target_counts[g] -= 1
                        current_sum -= 1

        # Feasibility check and draw
        selected = []
        for g, idxs in group_indices.items():
            k_g = target_counts[g]
            n_g = len(idxs)
            if k_g > n_g:
                raise ValueError(f"StratifiedSampler: requested {k_g} from group '{g}' with only {n_g} samples.")
            if k_g == 0:
                continue
            chosen = rng.choice(idxs, size=k_g, replace=False)
            selected.append(chosen)

        # It's possible all k_g == 0 if ratio is extremely small; enforce at least one globally.
        if len(selected) == 0:
            import warnings
            warnings.warn(
                f"StratifiedSampler: ratio={self.ratio} resulted in zero samples from all groups. "
                f"Falling back to selecting one sample from the largest group. "
                f"This violates strict stratification. Consider increasing the ratio.",
                UserWarning,
                stacklevel=4
            )
            # Choose from largest group to maintain stability.
            largest_group = max(group_indices.keys(), key=lambda gg: group_sizes[gg])
            chosen = rng.choice(group_indices[largest_group], size=1, replace=False)
            selected.append(chosen)

        idx = np.sort(np.concatenate(selected))
        return (idx,)


class ComplementaryPairsSampler(BaseSampler):
    """
    Complementary Pairs Stability Selection (CPSS) at π = 0.5 with drop-one for odd n.
    - Produces two disjoint index arrays per draw (half A and half B), each of size floor(n/2).
    - When n is odd, one sample is unused in each draw.
    """

    def draw(
        self,
        n: int,
        rng: np.random.Generator,
        labels: Optional[Sequence[str]] = None,
    ) -> Tuple[np.ndarray, ...]:
        perm = rng.permutation(n)
        half = n // 2
        a = np.sort(perm[:half])
        b = np.sort(perm[half:half + half])
        return (a, b)