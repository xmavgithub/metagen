"""
MetaGen Baselines

Provides baseline architectures and search strategies for comparison
in experiments. All baselines generate synthetic metrics deterministically.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import ClassVar

from metagen.specs.schema import ModelSpec


@dataclass
class BaselineMetrics:
    """Metrics produced by a baseline evaluation."""

    accuracy: float
    params_million: float
    latency_ms: float
    memory_mb: float
    flops_gflops: float
    training_hours: float
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "params_million": self.params_million,
            "latency_ms": self.latency_ms,
            "memory_mb": self.memory_mb,
            "flops_gflops": self.flops_gflops,
            "training_hours": self.training_hours,
            **self.extra,
        }


class Baseline(ABC):
    """Abstract base class for experiment baselines."""

    name: ClassVar[str] = "baseline"
    description: ClassVar[str] = "Abstract baseline"
    category: ClassVar[str] = "unknown"  # "manual", "search", "metagen"

    @abstractmethod
    def evaluate(self, spec: ModelSpec, seed: int) -> BaselineMetrics:
        """
        Evaluate the baseline on a given spec.

        Args:
            spec: Model specification to evaluate.
            seed: Random seed for deterministic evaluation.

        Returns:
            BaselineMetrics with synthetic scores.
        """
        pass

    def get_architecture_summary(self, spec: ModelSpec) -> dict:
        """Get architecture summary for this baseline."""
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
        }


# =============================================================================
# Manual Architecture Baselines
# =============================================================================


class GPT2Baseline(Baseline):
    """
    GPT-2 style baseline for text generation tasks.

    Represents a manually-designed decoder-only transformer
    following the GPT-2 architecture conventions.
    """

    name: ClassVar[str] = "gpt2"
    description: ClassVar[str] = "GPT-2 style decoder-only transformer"
    category: ClassVar[str] = "manual"

    # GPT-2 configurations
    CONFIGS: ClassVar[dict] = {
        "small": {"layers": 12, "hidden": 768, "heads": 12, "params": 117},
        "medium": {"layers": 24, "hidden": 1024, "heads": 16, "params": 345},
        "large": {"layers": 36, "hidden": 1280, "heads": 20, "params": 774},
        "xl": {"layers": 48, "hidden": 1600, "heads": 25, "params": 1557},
    }

    def __init__(self, size: str = "small"):
        """Initialize with a specific GPT-2 size."""
        if size not in self.CONFIGS:
            size = "small"
        self.size = size
        self.config = self.CONFIGS[size]

    def evaluate(self, spec: ModelSpec, seed: int) -> BaselineMetrics:
        """Evaluate GPT-2 baseline with synthetic metrics."""
        rnd = random.Random(seed)

        # Base accuracy depends on model size
        base_accuracy = 0.70 + (self.config["params"] / 2000) * 0.15
        accuracy = min(0.92, base_accuracy + rnd.uniform(-0.02, 0.03))

        # Compute efficiency metrics
        params = self.config["params"]
        latency = 10 + params * 0.02 + rnd.uniform(-2, 5)
        memory = params * 4 + rnd.uniform(-50, 100)  # ~4 bytes per param
        flops = params * 2 + rnd.uniform(-10, 20)
        training_hours = params * 0.1 + rnd.uniform(-5, 10)

        return BaselineMetrics(
            accuracy=round(accuracy, 4),
            params_million=params,
            latency_ms=round(latency, 2),
            memory_mb=round(memory, 1),
            flops_gflops=round(flops, 1),
            training_hours=round(max(1, training_hours), 1),
            extra={"config": self.size, "layers": self.config["layers"]},
        )

    def get_architecture_summary(self, spec: ModelSpec) -> dict:
        """Get GPT-2 architecture summary."""
        return {
            **super().get_architecture_summary(spec),
            "size": self.size,
            "layers": self.config["layers"],
            "hidden_size": self.config["hidden"],
            "attention_heads": self.config["heads"],
            "parameters_million": self.config["params"],
        }


class ResNetBaseline(Baseline):
    """
    ResNet baseline for image classification tasks.

    Represents manually-designed CNN architectures following
    the ResNet family conventions (ResNet-18 to ResNet-152).
    """

    name: ClassVar[str] = "resnet"
    description: ClassVar[str] = "ResNet convolutional neural network"
    category: ClassVar[str] = "manual"

    CONFIGS: ClassVar[dict] = {
        "18": {"layers": 18, "params": 11.7},
        "34": {"layers": 34, "params": 21.8},
        "50": {"layers": 50, "params": 25.6},
        "101": {"layers": 101, "params": 44.5},
        "152": {"layers": 152, "params": 60.2},
    }

    def __init__(self, depth: str = "50"):
        """Initialize with a specific ResNet depth."""
        if depth not in self.CONFIGS:
            depth = "50"
        self.depth = depth
        self.config = self.CONFIGS[depth]

    def evaluate(self, spec: ModelSpec, seed: int) -> BaselineMetrics:
        """Evaluate ResNet baseline with synthetic metrics."""
        rnd = random.Random(seed)

        # ImageNet-style accuracy scaling
        base_accuracy = 0.69 + (self.config["params"] / 100) * 0.08
        accuracy = min(0.85, base_accuracy + rnd.uniform(-0.01, 0.02))

        params = self.config["params"]
        latency = 5 + params * 0.3 + rnd.uniform(-1, 3)
        memory = params * 4 + rnd.uniform(-20, 50)
        flops = params * 4 + rnd.uniform(-5, 10)
        training_hours = params * 0.5 + rnd.uniform(-2, 5)

        return BaselineMetrics(
            accuracy=round(accuracy, 4),
            params_million=params,
            latency_ms=round(latency, 2),
            memory_mb=round(memory, 1),
            flops_gflops=round(flops, 1),
            training_hours=round(max(1, training_hours), 1),
            extra={"config": f"resnet{self.depth}", "layers": self.config["layers"]},
        )

    def get_architecture_summary(self, spec: ModelSpec) -> dict:
        """Get ResNet architecture summary."""
        return {
            **super().get_architecture_summary(spec),
            "depth": self.depth,
            "layers": self.config["layers"],
            "parameters_million": self.config["params"],
        }


class UNetBaseline(Baseline):
    """
    U-Net baseline for image generation/segmentation tasks.

    Represents manually-designed U-Net architectures commonly
    used in diffusion models and segmentation.
    """

    name: ClassVar[str] = "unet"
    description: ClassVar[str] = "U-Net encoder-decoder architecture"
    category: ClassVar[str] = "manual"

    CONFIGS: ClassVar[dict] = {
        "tiny": {"channels": [64, 128, 256], "params": 7.8},
        "small": {"channels": [64, 128, 256, 512], "params": 31.4},
        "base": {"channels": [128, 256, 512, 1024], "params": 125.5},
        "large": {"channels": [192, 384, 768, 1536], "params": 282.5},
    }

    def __init__(self, size: str = "base"):
        """Initialize with a specific U-Net size."""
        if size not in self.CONFIGS:
            size = "base"
        self.size = size
        self.config = self.CONFIGS[size]

    def evaluate(self, spec: ModelSpec, seed: int) -> BaselineMetrics:
        """Evaluate U-Net baseline with synthetic metrics."""
        rnd = random.Random(seed)

        # FID-style quality metric (lower is better, converted to 0-1 accuracy)
        base_quality = 0.75 + (self.config["params"] / 400) * 0.12
        accuracy = min(0.92, base_quality + rnd.uniform(-0.02, 0.03))

        params = self.config["params"]
        latency = 50 + params * 0.5 + rnd.uniform(-10, 30)
        memory = params * 5 + rnd.uniform(-100, 200)
        flops = params * 10 + rnd.uniform(-20, 50)
        training_hours = params * 0.2 + rnd.uniform(-5, 15)

        return BaselineMetrics(
            accuracy=round(accuracy, 4),
            params_million=params,
            latency_ms=round(latency, 2),
            memory_mb=round(memory, 1),
            flops_gflops=round(flops, 1),
            training_hours=round(max(1, training_hours), 1),
            extra={
                "config": self.size,
                "channels": self.config["channels"],
                "depth": len(self.config["channels"]),
            },
        )

    def get_architecture_summary(self, spec: ModelSpec) -> dict:
        """Get U-Net architecture summary."""
        return {
            **super().get_architecture_summary(spec),
            "size": self.size,
            "channel_progression": self.config["channels"],
            "parameters_million": self.config["params"],
        }


# =============================================================================
# Search Strategy Baselines
# =============================================================================


class RandomSearchBaseline(Baseline):
    """
    Random search baseline for architecture selection.

    Simulates randomly sampling architectures from the search space
    without any optimization strategy.
    """

    name: ClassVar[str] = "random_search"
    description: ClassVar[str] = "Random architecture sampling"
    category: ClassVar[str] = "search"

    def __init__(self, num_samples: int = 100):
        """Initialize with number of random samples."""
        self.num_samples = num_samples

    def evaluate(self, spec: ModelSpec, seed: int) -> BaselineMetrics:
        """Evaluate random search with synthetic metrics."""
        rnd = random.Random(seed)

        # Random search typically finds mediocre solutions
        base_accuracy = 0.65 + (self.num_samples / 1000) * 0.10
        accuracy = min(0.82, base_accuracy + rnd.uniform(-0.03, 0.05))

        # Random architectures have variable efficiency
        params = 50 + rnd.uniform(0, 200)
        latency = 20 + rnd.uniform(0, 80)
        memory = params * 4 + rnd.uniform(-50, 150)
        flops = params * 3 + rnd.uniform(-20, 60)
        training_hours = self.num_samples * 0.1 + rnd.uniform(-5, 10)

        return BaselineMetrics(
            accuracy=round(accuracy, 4),
            params_million=round(params, 1),
            latency_ms=round(latency, 2),
            memory_mb=round(memory, 1),
            flops_gflops=round(flops, 1),
            training_hours=round(max(1, training_hours), 1),
            extra={"samples_evaluated": self.num_samples},
        )

    def get_architecture_summary(self, spec: ModelSpec) -> dict:
        """Get random search summary."""
        return {
            **super().get_architecture_summary(spec),
            "num_samples": self.num_samples,
            "strategy": "uniform_random",
        }


class SingleObjectiveBaseline(Baseline):
    """
    Single-objective NAS baseline.

    Simulates traditional NAS that optimizes only for accuracy,
    without considering efficiency constraints.
    """

    name: ClassVar[str] = "single_objective"
    description: ClassVar[str] = "Single-objective NAS (accuracy only)"
    category: ClassVar[str] = "search"

    def __init__(self, objective: str = "accuracy", search_budget: int = 500):
        """Initialize with optimization objective."""
        self.objective = objective
        self.search_budget = search_budget

    def evaluate(self, spec: ModelSpec, seed: int) -> BaselineMetrics:
        """Evaluate single-objective NAS with synthetic metrics."""
        rnd = random.Random(seed)

        # Single-objective finds good accuracy but ignores efficiency
        base_accuracy = 0.80 + (self.search_budget / 2000) * 0.10
        accuracy = min(0.93, base_accuracy + rnd.uniform(-0.01, 0.03))

        # But efficiency is often poor (over-parameterized)
        params = 150 + rnd.uniform(50, 200)
        latency = 50 + rnd.uniform(20, 100)
        memory = params * 5 + rnd.uniform(0, 300)
        flops = params * 4 + rnd.uniform(0, 100)
        training_hours = self.search_budget * 0.05 + rnd.uniform(-3, 10)

        return BaselineMetrics(
            accuracy=round(accuracy, 4),
            params_million=round(params, 1),
            latency_ms=round(latency, 2),
            memory_mb=round(memory, 1),
            flops_gflops=round(flops, 1),
            training_hours=round(max(1, training_hours), 1),
            extra={
                "objective": self.objective,
                "search_budget": self.search_budget,
            },
        )

    def get_architecture_summary(self, spec: ModelSpec) -> dict:
        """Get single-objective NAS summary."""
        return {
            **super().get_architecture_summary(spec),
            "objective": self.objective,
            "search_budget": self.search_budget,
            "strategy": "evolutionary",
        }


# =============================================================================
# MetaGen Baseline (for comparison)
# =============================================================================


class MetaGenBaseline(Baseline):
    """
    MetaGen synthesis baseline.

    Represents the full MetaGen pipeline for comparison against
    manual architectures and simpler search strategies.
    """

    name: ClassVar[str] = "metagen"
    description: ClassVar[str] = "MetaGen multi-objective synthesis"
    category: ClassVar[str] = "metagen"

    def __init__(self, search_budget: int = 200):
        """Initialize with search configuration."""
        self.search_budget = search_budget

    def evaluate(self, spec: ModelSpec, seed: int) -> BaselineMetrics:
        """Evaluate MetaGen with synthetic metrics."""
        rnd = random.Random(seed)

        # MetaGen achieves good accuracy with balanced efficiency
        base_accuracy = 0.85 + (self.search_budget / 1000) * 0.08
        accuracy = min(0.95, base_accuracy + rnd.uniform(-0.01, 0.02))

        # Efficiency is optimized via multi-objective search
        params = 80 + rnd.uniform(-20, 40)
        latency = 25 + rnd.uniform(-5, 15)
        memory = params * 3.5 + rnd.uniform(-30, 50)
        flops = params * 2.5 + rnd.uniform(-10, 20)
        training_hours = self.search_budget * 0.02 + rnd.uniform(-2, 5)

        return BaselineMetrics(
            accuracy=round(accuracy, 4),
            params_million=round(params, 1),
            latency_ms=round(latency, 2),
            memory_mb=round(memory, 1),
            flops_gflops=round(flops, 1),
            training_hours=round(max(1, training_hours), 1),
            extra={
                "search_budget": self.search_budget,
                "pareto_solutions": rnd.randint(5, 15),
            },
        )

    def get_architecture_summary(self, spec: ModelSpec) -> dict:
        """Get MetaGen summary."""
        return {
            **super().get_architecture_summary(spec),
            "search_budget": self.search_budget,
            "strategy": "multi_objective_evolution",
            "objectives": ["accuracy", "params", "latency"],
        }


# =============================================================================
# Registry
# =============================================================================

_BASELINE_REGISTRY: dict[str, type[Baseline]] = {
    "gpt2": GPT2Baseline,
    "resnet": ResNetBaseline,
    "unet": UNetBaseline,
    "random_search": RandomSearchBaseline,
    "single_objective": SingleObjectiveBaseline,
    "metagen": MetaGenBaseline,
}


def get_baseline(name: str, **kwargs) -> Baseline:
    """
    Get a baseline by name.

    Args:
        name: Baseline name (e.g., "gpt2", "resnet", "random_search")
        **kwargs: Configuration options for the baseline

    Returns:
        Configured Baseline instance

    Example:
        >>> baseline = get_baseline("gpt2", size="large")
        >>> metrics = baseline.evaluate(spec, seed=42)
    """
    if name not in _BASELINE_REGISTRY:
        available = ", ".join(_BASELINE_REGISTRY.keys())
        raise ValueError(f"Unknown baseline: {name}. Available: {available}")
    return _BASELINE_REGISTRY[name](**kwargs)


def list_baselines() -> list[str]:
    """Return list of available baseline names."""
    return list(_BASELINE_REGISTRY.keys())
