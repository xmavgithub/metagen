"""
MetaGen Mutation Utilities

Shared mutation strategies for architecture search and refinement.
Provides consistent dimension mutation with configurable exploration vs exploitation.
"""

from __future__ import annotations

import random
from typing import Literal

# Mutation range constants
EXPLORATION_RANGE = (0.75, 1.25)  # For search engine - wider exploration
EXPLOITATION_RANGE = (0.85, 1.15)  # For refiner - fine-tuning

# Layer mutation deltas
LAYER_DELTAS = (-2, -1, 1, 2)
HEAD_DELTAS = (-2, -1, 1, 2)

# Minimum values
MIN_HIDDEN_SIZE = 64
MIN_LAYERS = 1
MIN_HEADS = 1

# Hidden size alignment
HIDDEN_SIZE_ALIGNMENT = 64


def mutate_dims(
    dims: dict[str, int],
    rng: random.Random,
    *,
    mutation_rate: float = 0.3,
    family: str = "transformer",
    mode: Literal["exploration", "exploitation"] = "exploration",
) -> dict[str, int]:
    """Mutate architecture dimensions with configurable exploration/exploitation tradeoff.

    Args:
        dims: Current dimensions (hidden_size, layers, heads).
        rng: Seeded random instance for reproducibility.
        mutation_rate: Probability of mutating each dimension.
        family: Architecture family (affects head adjustment for CNNs).
        mode: "exploration" for wider search, "exploitation" for refinement.

    Returns:
        New dimension dict with potentially mutated values.

    Example:
        >>> rng = random.Random(42)
        >>> dims = {"hidden_size": 512, "layers": 12, "heads": 8}
        >>> new_dims = mutate_dims(dims, rng, mode="exploitation")
        >>> new_dims["hidden_size"] % 64 == 0
        True
    """
    hidden_range = EXPLOITATION_RANGE if mode == "exploitation" else EXPLORATION_RANGE

    hidden_size = dims["hidden_size"]
    layers = dims["layers"]
    heads = dims["heads"]

    # Apply mutations based on mutation rate
    if rng.random() < mutation_rate:
        scale = rng.uniform(*hidden_range)
        scaled = round(hidden_size * scale / HIDDEN_SIZE_ALIGNMENT) * HIDDEN_SIZE_ALIGNMENT
        hidden_size = int(max(MIN_HIDDEN_SIZE, scaled))

    if rng.random() < mutation_rate:
        layers = max(MIN_LAYERS, layers + rng.choice(LAYER_DELTAS))

    if rng.random() < mutation_rate:
        heads = max(MIN_HEADS, heads + rng.choice(HEAD_DELTAS))

    # Ensure heads divides hidden_size evenly
    while heads > 1 and hidden_size % heads != 0:
        heads -= 1

    # CNN adjustment: reduce heads
    if family == "cnn":
        heads = max(MIN_HEADS, heads // 2)

    return {"hidden_size": hidden_size, "layers": layers, "heads": heads}


def sample_dims(
    base_dims: dict[str, int],
    rng: random.Random,
    *,
    family: str = "transformer",
) -> dict[str, int]:
    """Sample new dimensions based on base dimensions.

    Uses wider discrete sampling for initial population generation.

    Args:
        base_dims: Base dimensions to sample around.
        rng: Seeded random instance.
        family: Architecture family.

    Returns:
        Sampled dimension dict.
    """
    hidden_size = base_dims["hidden_size"]
    layers = base_dims["layers"]

    # Use discrete factors for more varied exploration
    hidden_factor = rng.choice([0.5, 0.75, 1.0, 1.25, 1.5])
    scaled = round(hidden_size * hidden_factor / HIDDEN_SIZE_ALIGNMENT) * HIDDEN_SIZE_ALIGNMENT
    hidden_size = int(max(MIN_HIDDEN_SIZE, scaled))

    layer_delta = rng.randint(-4, 4)
    layers = max(MIN_LAYERS, layers + layer_delta)

    heads = max(MIN_HEADS, hidden_size // HIDDEN_SIZE_ALIGNMENT)
    while heads > 1 and hidden_size % heads != 0:
        heads -= 1

    if family == "cnn":
        heads = max(MIN_HEADS, heads // 2)

    return {"hidden_size": hidden_size, "layers": layers, "heads": heads}
