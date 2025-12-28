"""
MetaGen Time Series Task Handlers

This module provides task handlers for time series forecasting and anomaly detection.

Supported modalities:
- time_series: Temporal sequences
- tabular: Structured numeric time series
- multimodal: Mixed inputs with temporal components

Example spec:
    metagen_version: "1.0"
    name: "stock_forecaster"
    modality:
      inputs: [time_series]
      outputs: [time_series]
    task:
      type: time_series_forecast
      domain: finance
      horizon: 30
      lookback: 90
      num_outputs: 1
    architecture:
      family: transformer

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

DEFAULT_LOOKBACK = 30
DEFAULT_HORIZON = 1
DEFAULT_OUTPUT_DIM = 1


class BaseTimeSeriesTaskHandler(TaskHandler):
    """Shared implementation for time series task handlers."""

    task_type: str = "time_series_forecast"
    metrics: tuple[str, ...] = ("mse", "rmse", "mae", "mape", "smape")
    loss_type: str = "mse"
    loss_fragment: str = "losses/mse_loss.py.j2"

    @property
    def name(self) -> str:
        return self.task_type

    @property
    def supported_modalities(self) -> list[str]:
        return ["time_series", "tabular", "multimodal"]

    @property
    def output_type(self) -> str:
        return "time_series"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add time-series-specific parameters to blueprint."""
        horizon = spec.task.horizon or DEFAULT_HORIZON
        lookback = spec.task.lookback or DEFAULT_LOOKBACK
        num_outputs = spec.task.num_outputs or DEFAULT_OUTPUT_DIM

        return replace(
            blueprint,
            horizon=horizon,
            lookback=lookback,
            num_outputs=num_outputs,
        )

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define time series head architecture."""
        hidden_size = blueprint.dims["hidden_size"]
        horizon = blueprint.horizon or spec.task.horizon or DEFAULT_HORIZON
        output_dim = blueprint.num_outputs or spec.task.num_outputs or DEFAULT_OUTPUT_DIM

        return {
            "type": "time_series_head",
            "hidden_dim": hidden_size,
            "horizon": horizon,
            "output_dim": output_dim,
            "dropout": 0.1,
            "pooling": "last",
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return time series loss function."""
        return self.loss_type

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return evaluation metrics for time series."""
        return list(self.metrics)

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get time series template fragments."""
        return [
            "heads/time_series_head.py.j2",
            self.loss_fragment,
            "data/time_series_datasets.py.j2",
        ]


@register_task("time_series_forecast")
class TimeSeriesForecastTaskHandler(BaseTimeSeriesTaskHandler):
    """Task handler for time series forecasting."""

    task_type = "time_series_forecast"
    metrics = ("mse", "rmse", "mae", "mape", "smape")
    loss_type = "mse"
    loss_fragment = "losses/mse_loss.py.j2"


@register_task("anomaly_detection")
class AnomalyDetectionTaskHandler(BaseTimeSeriesTaskHandler):
    """Task handler for time series anomaly detection."""

    task_type = "anomaly_detection"
    metrics = ("roc_auc", "pr_auc", "f1", "precision", "recall")
    loss_type = "mse"
    loss_fragment = "losses/reconstruction.py.j2"

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Default anomaly detection to reconstruct the lookback window."""
        lookback = spec.task.lookback or DEFAULT_LOOKBACK
        horizon = spec.task.horizon or lookback
        num_outputs = spec.task.num_outputs or DEFAULT_OUTPUT_DIM

        return replace(
            blueprint,
            horizon=horizon,
            lookback=lookback,
            num_outputs=num_outputs,
        )
