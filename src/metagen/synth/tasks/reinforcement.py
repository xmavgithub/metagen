"""
MetaGen Reinforcement Learning Task Handlers

This module provides task handlers for reinforcement learning tasks:
policy gradient, value-based, actor-critic, and model-based methods.

Supported modalities:
- image: Visual observations
- text: Text-based environments
- tabular: Low-dimensional state vectors
- time_series: Temporal state sequences
- multimodal: Mixed observations

Example spec:
    metagen_version: "1.0"
    name: "atari_agent"
    modality:
      inputs: [image]
      outputs: [action]
    task:
      type: policy_gradient
      domain: game
      action_space: discrete
      num_actions: 18
    architecture:
      family: cnn

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

DEFAULT_NUM_ACTIONS = {
    "atari": 18,
    "game": 10,
    "robotics": 12,
    "generic": 6,
}
DEFAULT_ACTION_DIM = 4
DEFAULT_ACTION_SPACE = "discrete"


class BaseRLTaskHandler(TaskHandler):
    """Shared implementation for reinforcement learning task handlers."""

    task_type: str = "policy_gradient"
    head_type: str = "rl_policy_head"
    loss_type: str = "policy_gradient_loss"
    metrics: tuple[str, ...] = ("episode_return", "episode_length", "entropy")

    @property
    def name(self) -> str:
        return self.task_type

    @property
    def supported_modalities(self) -> list[str]:
        return ["image", "text", "tabular", "time_series", "multimodal"]

    @property
    def output_type(self) -> str:
        return "action"

    def _resolve_action_spec(
        self,
        spec: ModelSpec,
    ) -> tuple[str, int | None, int | None]:
        action_space = (spec.task.action_space or DEFAULT_ACTION_SPACE).lower()
        if action_space not in {"discrete", "continuous"}:
            logger.warning("Unknown action_space '%s', defaulting to discrete.", action_space)
            action_space = "discrete"

        num_actions = spec.task.num_actions
        action_dim = spec.task.action_dim

        if action_space == "discrete":
            if num_actions is None:
                domain = (spec.task.domain or "").lower()
                num_actions = DEFAULT_NUM_ACTIONS.get(domain, DEFAULT_NUM_ACTIONS["generic"])
        else:
            if action_dim is None:
                action_dim = DEFAULT_ACTION_DIM

        return action_space, num_actions, action_dim

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """Add RL-specific parameters to blueprint."""
        action_space, num_actions, action_dim = self._resolve_action_spec(spec)
        return replace(
            blueprint,
            action_space=action_space,
            num_actions=num_actions,
            action_dim=action_dim,
        )

    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """Define RL head architecture."""
        action_space = (blueprint.action_space or DEFAULT_ACTION_SPACE).lower()
        num_actions = blueprint.num_actions
        action_dim = blueprint.action_dim
        if action_space == "discrete" and num_actions is None:
            action_space, num_actions, action_dim = self._resolve_action_spec(spec)
        elif action_space == "continuous" and action_dim is None:
            action_space, num_actions, action_dim = self._resolve_action_spec(spec)
        return {
            "type": self.head_type,
            "hidden_dim": blueprint.dims["hidden_size"],
            "action_space": action_space,
            "num_actions": num_actions,
            "action_dim": action_dim,
            "dropout": 0.1,
        }

    def get_loss_function(self, spec: ModelSpec) -> str:
        """Return RL loss function."""
        return self.loss_type

    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """Return RL evaluation metrics."""
        return list(self.metrics)

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """Get RL template fragments."""
        return [
            f"heads/{self.head_type}.py.j2",
            "losses/rl_loss.py.j2",
            "data/rl_environments.py.j2",
        ]


@register_task("policy_gradient")
class PolicyGradientTaskHandler(BaseRLTaskHandler):
    """Task handler for policy gradient methods (PPO, A3C, REINFORCE)."""

    task_type = "policy_gradient"
    head_type = "rl_policy_head"
    loss_type = "policy_gradient_loss"
    metrics = ("episode_return", "episode_length", "entropy", "kl_divergence")


@register_task("value_based")
class ValueBasedTaskHandler(BaseRLTaskHandler):
    """Task handler for value-based methods (DQN, Rainbow)."""

    task_type = "value_based"
    head_type = "rl_value_head"
    loss_type = "td_loss"
    metrics = ("avg_q_value", "td_error", "episode_return")


@register_task("actor_critic")
class ActorCriticTaskHandler(BaseRLTaskHandler):
    """Task handler for actor-critic methods (SAC, TD3)."""

    task_type = "actor_critic"
    head_type = "rl_actor_critic_head"
    loss_type = "actor_critic_loss"
    metrics = ("episode_return", "value_loss", "policy_loss", "entropy")


@register_task("model_based")
class ModelBasedTaskHandler(BaseRLTaskHandler):
    """Task handler for model-based RL (world models, MuZero-style)."""

    task_type = "model_based"
    head_type = "rl_model_head"
    loss_type = "model_based_loss"
    metrics = ("episode_return", "dynamics_loss", "policy_loss")
