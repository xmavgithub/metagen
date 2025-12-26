"""
MetaGen AutoML Training Utilities

Shared utilities for prototype training across search engine and refiner modules.
This module eliminates code duplication for training-related operations.

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from metagen.synth import architecture

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec

logger = logging.getLogger(__name__)


def maybe_train_prototype(
    *,
    metrics: dict[str, float],
    score: float,
    dims: dict[str, int],
    spec: ModelSpec,
    train_prototypes: bool,
    prototype_steps: int,
    rng: random.Random,
    loss_weight: float = 0.1,
) -> tuple[dict[str, float], float]:
    """
    Optionally train a prototype model and update metrics/score.

    This is a shared utility used by both the search engine and refiner
    to train small prototype models and incorporate training loss into
    the architecture evaluation.

    Args:
        metrics: Current metrics dictionary (will be copied, not modified).
        score: Current candidate score.
        dims: Architecture dimensions dict with hidden_size, layers, heads.
        spec: Model specification for synthesis.
        train_prototypes: If False, returns inputs unchanged.
        prototype_steps: Number of training steps for prototype.
        rng: Random number generator for seed derivation.
        loss_weight: Weight for prototype loss in score adjustment (default 0.1).

    Returns:
        Tuple of (updated_metrics, updated_score).
        If training fails, prototype_loss is set to inf.

    Example:
        >>> metrics, score = maybe_train_prototype(
        ...     metrics={"params": 1e9},
        ...     score=0.8,
        ...     dims={"hidden_size": 768, "layers": 12, "heads": 12},
        ...     spec=spec,
        ...     train_prototypes=True,
        ...     prototype_steps=100,
        ...     rng=random.Random(42),
        ... )
        >>> print(metrics.get("prototype_loss"))
        0.523
    """
    if not train_prototypes:
        return metrics, score

    # Import here to avoid circular dependencies
    from metagen.automl.prototype_trainer import PrototypeTrainer

    try:
        trainer = PrototypeTrainer()
        blueprint = architecture.build_blueprint_from_dims(
            spec,
            dims,
            seed=rng.randint(0, 2**32 - 1),
        )
        train_metrics = trainer.train_prototype(
            blueprint,
            spec,
            budget_steps=prototype_steps,
        )
        # Create new dict to avoid mutating input
        updated_metrics = dict(metrics)
        updated_metrics["prototype_loss"] = train_metrics.final_loss
        updated_metrics["prototype_steps"] = float(train_metrics.steps)
        updated_metrics["prototype_runtime_sec"] = train_metrics.runtime_sec
        updated_score = score - train_metrics.final_loss * loss_weight
        return updated_metrics, updated_score
    except Exception as exc:
        logger.warning("Prototype training failed: %s", exc)
        updated_metrics = dict(metrics)
        updated_metrics["prototype_loss"] = float("inf")
        updated_metrics["prototype_steps"] = 0.0
        updated_metrics["prototype_runtime_sec"] = 0.0
        return updated_metrics, score
