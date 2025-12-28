"""
MetaGen Task Handlers - Base Classes

This module provides the abstract base classes for task-specific synthesis
handlers. Task handlers complement modality handlers by adding task-specific
logic for classification, detection, segmentation, RL, and other task types.

The task handler pattern allows MetaGen to support multiple task types with
clean separation of concerns. Each handler is responsible for:

1. Augmenting BlueprintState with task-specific parameters (num_classes, etc.)
2. Defining the output head architecture (classification head, detection head)
3. Specifying appropriate loss functions and evaluation metrics
4. Selecting task-specific template fragments for code generation

Architecture:
    TaskHandler (ABC)
    └── ClassificationTaskHandler  - Classification tasks
    └── RegressionTaskHandler      - Regression tasks
    └── DetectionTaskHandler       - Object detection
    └── SegmentationTaskHandler    - Semantic/instance segmentation
    └── TimeSeriesTaskHandler      - Time series forecasting
    └── RLTaskHandler              - Reinforcement learning
    └── GraphTaskHandler           - Graph neural networks

Example Usage:
    >>> from metagen.synth.tasks import get_task_handler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/image/image_classifier_resnet.yaml")
    >>> handler = get_task_handler(spec)
    >>> print(handler.name)
    'classification'
    >>>
    >>> # Augment blueprint with task-specific parameters
    >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
    >>> print(augmented.num_classes)
    1000
    >>>
    >>> # Get task-specific head architecture
    >>> head = handler.get_head_architecture(spec, augmented)
    >>> print(head['type'])
    'classification_head'

Author: MetaGen Team
Created: 2025-12-28
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskComponents:
    """
    Container for task-specific code generation components.

    This dataclass holds the information needed by the code generator
    to produce task-appropriate model code. Each field specifies
    which template fragments or code patterns to use for the output
    head and training configuration.

    Attributes:
        head_type: Type of output head to generate. Valid values:
            - "classification_head": Softmax classification
            - "regression_head": Linear regression output
            - "detection_head": Bounding box + class predictions
            - "segmentation_head": Per-pixel classification
            - "embedding_head": Vector embedding output
            - "rl_policy_head": RL policy network head
            - "rl_value_head": RL value network head
        loss_type: Loss function to use for training.
            Examples: "cross_entropy", "mse", "focal", "detection_loss"
        metrics: Evaluation metrics for this task.
            Examples: ["accuracy", "f1"], ["mAP", "mAP_50"], ["mse", "mae"]
        template_fragments: List of template fragment paths to include.
        additional_imports: Extra Python imports needed for this task.
        config: Additional task-specific configuration as a dict.
    """

    head_type: str
    loss_type: str = "cross_entropy"
    metrics: tuple[str, ...] = field(default_factory=lambda: ("accuracy",))
    template_fragments: tuple[str, ...] = field(default_factory=tuple)
    additional_imports: tuple[str, ...] = field(default_factory=tuple)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize components after initialization."""
        if isinstance(self.metrics, list):
            object.__setattr__(self, "metrics", tuple(self.metrics))
        if isinstance(self.template_fragments, list):
            object.__setattr__(self, "template_fragments", tuple(self.template_fragments))
        if isinstance(self.additional_imports, list):
            object.__setattr__(self, "additional_imports", tuple(self.additional_imports))

    def with_config(self, **kwargs: Any) -> TaskComponents:
        """Create a new TaskComponents with updated config values."""
        new_config = {**self.config, **kwargs}
        return TaskComponents(
            head_type=self.head_type,
            loss_type=self.loss_type,
            metrics=self.metrics,
            template_fragments=self.template_fragments,
            additional_imports=self.additional_imports,
            config=new_config,
        )


class TaskHandler(ABC):
    """
    Abstract base class for task-specific synthesis handlers.

    A TaskHandler encapsulates all task-specific logic for the MetaGen
    synthesis pipeline. This includes:

    1. **Blueprint Augmentation**: Adding task-specific parameters to the
       BlueprintState (e.g., num_classes for classification)

    2. **Head Architecture**: Defining the output head structure
       (classification head, detection head, etc.)

    3. **Loss Functions**: Specifying appropriate training objectives

    4. **Metrics**: Defining evaluation metrics for the task

    5. **Template Selection**: Choosing correct code templates

    Task handlers work alongside modality handlers:
    - Modality handlers define the backbone/encoder (what processes input)
    - Task handlers define the head/decoder (what produces output)

    Subclasses must implement all abstract methods to support a specific
    task type.

    Attributes:
        name: Task type name (e.g., "classification", "detection").
        supported_modalities: List of input modalities this task supports.
        output_type: The output type produced by this task.

    Example:
        >>> class ClassificationTaskHandler(TaskHandler):
        ...     @property
        ...     def name(self) -> str:
        ...         return "classification"
        ...
        ...     @property
        ...     def supported_modalities(self) -> list[str]:
        ...         return ["image", "text", "audio", "tabular"]
        ...
        ...     @property
        ...     def output_type(self) -> str:
        ...         return "label"
        ...
        ...     # ... implement other abstract methods
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Task type name.

        Returns:
            Task name as lowercase string (e.g., "classification", "detection").
        """
        ...

    @property
    @abstractmethod
    def supported_modalities(self) -> list[str]:
        """
        List of input modalities this task handler supports.

        Returns:
            List of modality names (e.g., ["image", "text", "audio"]).
        """
        ...

    @property
    @abstractmethod
    def output_type(self) -> str:
        """
        The output type produced by this task.

        Returns:
            Output type string (e.g., "label", "bounding_boxes", "embedding").
        """
        ...

    @abstractmethod
    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """
        Augment blueprint with task-specific parameters.

        This method adds parameters specific to this task type to the
        BlueprintState. For example:
        - Classification: num_classes
        - Detection: num_classes, num_anchors
        - Time Series: horizon, lookback
        - RL: num_actions, action_dim

        Args:
            spec: The model specification.
            blueprint: The base BlueprintState with modality-specific params.
            seed: Random seed for deterministic augmentation.

        Returns:
            New BlueprintState with task-specific parameters added.

        Example:
            >>> handler = ClassificationTaskHandler()
            >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
            >>> print(augmented.num_classes)
            1000
        """
        ...

    @abstractmethod
    def get_head_architecture(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> dict[str, Any]:
        """
        Define the task-specific output head architecture.

        Returns a dictionary describing the output head configuration,
        which is used by code generation to create the appropriate
        PyTorch module.

        Args:
            spec: The model specification.
            blueprint: The augmented BlueprintState.

        Returns:
            Dictionary with head architecture parameters.

        Example:
            >>> head = handler.get_head_architecture(spec, blueprint)
            >>> print(head)
            {
                'type': 'classification_head',
                'hidden_dim': 768,
                'num_classes': 1000,
                'dropout': 0.1,
                'activation': 'gelu'
            }
        """
        ...

    @abstractmethod
    def get_loss_function(self, spec: ModelSpec) -> str:
        """
        Return appropriate loss function for this task.

        Args:
            spec: The model specification.

        Returns:
            Loss function name (e.g., "cross_entropy", "focal", "mse").
        """
        ...

    @abstractmethod
    def get_metrics(self, spec: ModelSpec) -> list[str]:
        """
        Return evaluation metrics for this task.

        Args:
            spec: The model specification.

        Returns:
            List of metric names.

        Example:
            >>> handler = ClassificationTaskHandler()
            >>> handler.get_metrics(spec)
            ['accuracy', 'top5_accuracy', 'f1_macro']
        """
        ...

    def generate_components(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> TaskComponents:
        """
        Generate task-specific code components.

        This method creates a TaskComponents instance with all the
        information needed for code generation.

        Args:
            spec: The model specification.
            blueprint: The augmented BlueprintState.
            seed: Random seed for deterministic generation.

        Returns:
            TaskComponents instance specifying code components.
        """
        head = self.get_head_architecture(spec, blueprint)
        return TaskComponents(
            head_type=head.get("type", f"{self.name}_head"),
            loss_type=self.get_loss_function(spec),
            metrics=tuple(self.get_metrics(spec)),
            template_fragments=tuple(self.get_template_fragments(spec, blueprint)),
            config=head,
        )

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """
        Get list of template fragments for this task.

        Returns paths to Jinja2 templates that should be included
        when generating code. Override in subclasses for task-specific
        templates.

        Args:
            spec: The model specification.
            blueprint: The augmented BlueprintState.

        Returns:
            List of template fragment paths.
        """
        return [f"heads/{self.name}_head.py.j2"]

    def validate_spec(self, spec: ModelSpec) -> None:
        """
        Validate that a spec is compatible with this task handler.

        Checks that:
        1. The spec's primary input modality is supported
        2. Required task-specific fields are present

        Args:
            spec: The model specification to validate.

        Raises:
            ValueError: If spec is invalid or incompatible.
        """
        if not spec.modality.inputs:
            raise ValueError("Spec has no input modalities defined")

        primary_input = spec.modality.inputs[0].lower()
        if primary_input not in self.supported_modalities:
            supported = ", ".join(self.supported_modalities)
            raise ValueError(
                f"Input modality '{primary_input}' is not supported by "
                f"{self.name} task handler. Supported modalities: {supported}"
            )

        logger.debug(
            f"Validated spec for {self.name} task handler: "
            f"inputs={spec.modality.inputs}, task_type={spec.task.type}"
        )

    def supports_spec(self, spec: ModelSpec) -> bool:
        """
        Check if this handler can process the given spec.

        Args:
            spec: The model specification to check.

        Returns:
            True if this handler can process the spec, False otherwise.
        """
        try:
            self.validate_spec(spec)
            return True
        except ValueError:
            return False

    def __repr__(self) -> str:
        """Return string representation of handler."""
        return f"{self.__class__.__name__}(name='{self.name}')"
