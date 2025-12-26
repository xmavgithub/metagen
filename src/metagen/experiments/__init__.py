"""
MetaGen Experiments Module

Provides infrastructure for running reproducible experiments comparing
MetaGen synthesis against baseline architectures and search strategies.
"""

from metagen.experiments.ablations import (
    AblationResult,
    AblationStudy,
    run_ablation,
)
from metagen.experiments.baselines import (
    Baseline,
    GPT2Baseline,
    RandomSearchBaseline,
    ResNetBaseline,
    SingleObjectiveBaseline,
    UNetBaseline,
    get_baseline,
    list_baselines,
)
from metagen.experiments.runner import (
    ExperimentResult,
    ExperimentRunner,
    run_experiment,
)

__all__ = [
    # Baselines
    "Baseline",
    "GPT2Baseline",
    "ResNetBaseline",
    "UNetBaseline",
    "RandomSearchBaseline",
    "SingleObjectiveBaseline",
    "get_baseline",
    "list_baselines",
    # Runner
    "ExperimentRunner",
    "ExperimentResult",
    "run_experiment",
    # Ablations
    "AblationStudy",
    "AblationResult",
    "run_ablation",
]
