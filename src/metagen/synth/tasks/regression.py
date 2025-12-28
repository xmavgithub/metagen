"""
MetaGen Regression Task Handler

This module provides the task handler for regression tasks, including
numeric prediction, continuous value estimation, and multi-output regression.

Supported modalities:
- tabular: MLP, transformer-based regressors
- image: CNN, ViT-based regressors (e.g., age estimation, pose estimation)
- text: Transformer-based regressors (e.g., sentiment scores, readability)
- time_series: Temporal regressors

Example spec:
    metagen_version: "1.0"
    name: "house_price_predictor"
    modality:
      inputs: [tabular]
      outputs: [regression]
    task:
      type: regression
      domain: finance
      num_outputs: 1
    architecture:
      family: mlp

Author: MetaGen Team
Created: 2025-12-28
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from metagen.synth.tasks.base import TaskHandler
from metagen.synth.tasks.registry import register_task

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState

logger = logging.getLogger(__name__)


@register_task("regression")
class RegressionTaskHandler(TaskHandler):
    """
    Task handler for regression tasks.

    Supports tabular, image, text, and time series regression with various
    architecture families (mlp, transformer, cnn).

    The handler:
    - Augments BlueprintState with num_outputs
    - Generates regression head architecture
    - Specifies MSE/MAE loss and regression metrics
    """

    @property
    def name(self) -> str:
        return "regression"

    @property
    def supported_modalities(self) -> list[str]:
        return ["tabular", "image", "text", "time_series", "multimodal"]

    @property
    def output_type(self) -> str:
        return "regression"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add regression-specific parameters to blueprint."""
        num_outputs = spec.task.num_outputs or 1
        return replace(blueprint, num_outputs=num_outputs)

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define regression head architecture."""
        hidden_size = blueprint.dims["hidden_size"]
        num_outputs = blueprint.num_outputs or 1

        # Multi-output regression may need larger intermediate layers
        if num_outputs > 10:
            intermediate_dim = hidden_size
            num_layers = 2
        else:
            intermediate_dim = hidden_size // 2
            num_layers = 1

        return {
            "type": "regression_head",
            "hidden_dim": hidden_size,
            "intermediate_dim": intermediate_dim,
            "num_outputs": num_outputs,
            "num_layers": num_layers,
            "dropout": 0.1,
            "activation": "relu",
            "output_activation": None,  # Linear output for unbounded regression
            "use_batch_norm": True,
            "pooling": "cls_token" if blueprint.family == "transformer" else "global_avg",
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return regression loss function."""
        # MSE is the default for regression
        # Could extend to support Huber, MAE based on spec
        return "mse"

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return regression evaluation metrics."""
        return [
            "mse",
            "rmse",
            "mae",
            "r2_score",
            "explained_variance",
        ]

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get regression template fragments."""
        fragments = ["heads/regression_head.py.j2"]

        # Add modality-specific data loader template
        primary_input = spec.modality.inputs[0].lower()
        if primary_input == "tabular":
            fragments.append("data/tabular_regression_dataset.py.j2")
        elif primary_input == "image":
            fragments.append("data/image_regression_dataset.py.j2")

        # Add loss template
        fragments.append("losses/mse_loss.py.j2")

        return fragments
