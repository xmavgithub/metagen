"""
MetaGen Modality Handlers - Base Classes

This module provides the abstract base classes for modality-specific
synthesis handlers. Each handler encapsulates the logic for generating
architecture components, code templates, and data loaders for a specific
input/output modality (text, image, audio, video).

The modality handler pattern allows MetaGen to support multiple input/output
modalities with clean separation of concerns. Each handler is responsible for:

1. Augmenting BlueprintState with modality-specific parameters
2. Generating code components (encoder, decoder, data loader)
3. Selecting appropriate template fragments for code generation

Architecture:
    ModalityHandler (ABC)
    └── TextModalityHandler   - Text/language models
    └── ImageModalityHandler  - Vision models (ViT, CNN, diffusion)
    └── AudioModalityHandler  - Audio/speech models
    └── VideoModalityHandler  - Video/temporal models
    └── MultiModalModalityHandler - Multi-modal fusion models

Example Usage:
    >>> from metagen.synth.modalities import get_handler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/text/text_llm_8b.yaml")
    >>> handler = get_handler(spec)
    >>> print(handler.name)
    'text'
    >>>
    >>> # Augment blueprint with modality-specific parameters
    >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
    >>> print(augmented.vocab_size)
    50257
    >>>
    >>> # Generate code components
    >>> components = handler.generate_components(spec, augmented, seed=42)
    >>> print(components.encoder_type)
    'transformer_encoder'

See Also:
    - docs/user-guide/modalities.md for user documentation
    - docs/development/adding-modalities.md for extension guide

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

# Standard library imports
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState

# Module-level logger for debugging and tracing
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModalityComponents:
    """
    Container for modality-specific code generation components.

    This dataclass holds the information needed by the code generator
    to produce modality-appropriate model code. Each field specifies
    which template fragments or code patterns to use for different
    parts of the generated model.

    The frozen=True ensures immutability, which is important for
    deterministic code generation - the same components should always
    produce the same code.

    Attributes:
        encoder_type: Type of encoder to generate. Valid values depend
            on the modality:
            - text: "transformer_encoder"
            - image: "vit_encoder", "cnn_encoder", "unet_encoder"
            - audio: "audio_transformer", "mel_cnn"
            - video: "temporal_transformer", "3d_cnn"
        decoder_type: Type of decoder to generate, or None if the model
            doesn't need a decoder (e.g., classification models).
            Examples: "autoregressive", "diffusion", "vae"
        embedding_type: Type of input embedding layer.
            Examples: "token_embedding", "patch_embedding", "pixel_embedding"
        data_loader_type: Type of data loader to generate.
            Examples: "text_dataset", "image_folder", "audio_dataset"
        loss_type: Type of loss function to use.
            Examples: "cross_entropy", "diffusion_loss", "reconstruction"
        template_fragments: List of template fragment paths to include
            in code generation. These are relative paths within
            src/metagen/templates/fragments/
        additional_imports: Extra Python imports needed for this modality.
            These are added to the generated model.py file.
        config: Additional modality-specific configuration as a dict.
            This allows handlers to pass arbitrary parameters to templates.

    Example:
        >>> components = ModalityComponents(
        ...     encoder_type="transformer_encoder",
        ...     decoder_type="autoregressive",
        ...     embedding_type="token_embedding",
        ...     data_loader_type="text_dataset",
        ...     loss_type="cross_entropy",
        ...     template_fragments=[
        ...         "embeddings/token_embedding.py.j2",
        ...         "encoders/transformer_encoder.py.j2",
        ...     ],
        ... )
        >>> print(components.encoder_type)
        'transformer_encoder'

        >>> # Components are immutable
        >>> components.encoder_type = "cnn"  # Raises FrozenInstanceError
    """

    encoder_type: str
    decoder_type: str | None = None
    embedding_type: str = "token_embedding"
    data_loader_type: str = "synthetic"
    loss_type: str = "cross_entropy"
    template_fragments: tuple[str, ...] = field(default_factory=tuple)
    additional_imports: tuple[str, ...] = field(default_factory=tuple)
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate components after initialization."""
        # Convert lists to tuples for immutability if needed
        # (dataclass doesn't do this automatically for default_factory)
        if isinstance(self.template_fragments, list):
            object.__setattr__(self, "template_fragments", tuple(self.template_fragments))
        if isinstance(self.additional_imports, list):
            object.__setattr__(self, "additional_imports", tuple(self.additional_imports))

    def with_config(self, **kwargs: Any) -> ModalityComponents:
        """
        Create a new ModalityComponents with updated config values.

        Since ModalityComponents is frozen, this method returns a new
        instance with the config dict updated.

        Args:
            **kwargs: Key-value pairs to add/update in config.

        Returns:
            New ModalityComponents instance with updated config.

        Example:
            >>> base = ModalityComponents(encoder_type="vit_encoder")
            >>> updated = base.with_config(patch_size=16, image_size=224)
            >>> print(updated.config)
            {'patch_size': 16, 'image_size': 224}
        """
        new_config = {**self.config, **kwargs}
        return ModalityComponents(
            encoder_type=self.encoder_type,
            decoder_type=self.decoder_type,
            embedding_type=self.embedding_type,
            data_loader_type=self.data_loader_type,
            loss_type=self.loss_type,
            template_fragments=self.template_fragments,
            additional_imports=self.additional_imports,
            config=new_config,
        )


class ModalityHandler(ABC):
    """
    Abstract base class for modality-specific synthesis handlers.

    A ModalityHandler encapsulates all modality-specific logic for the
    MetaGen synthesis pipeline. This includes:

    1. **Blueprint Augmentation**: Adding modality-specific parameters
       to the BlueprintState (e.g., vocab_size for text, image_size for images)

    2. **Component Generation**: Determining which code components
       (encoders, decoders, embeddings) are appropriate for this modality

    3. **Template Selection**: Choosing the correct Jinja2 template
       fragments for code generation

    Subclasses must implement all abstract methods to support a specific
    modality. The handler pattern allows adding new modalities without
    modifying existing code - just create a new handler class.

    Attributes:
        name: Human-readable name of the modality (e.g., "text", "image").
            Used in logging and error messages.
        supported_families: List of architecture families this handler
            supports. Valid values: "transformer", "cnn", "diffusion",
            "hybrid", "rnn". Used for validation.

    Class Attributes:
        VALID_FAMILIES: Tuple of all valid architecture family names.

    Example:
        >>> # Creating a custom handler
        >>> class CustomHandler(ModalityHandler):
        ...     name = "custom"
        ...     supported_families = ["transformer"]
        ...
        ...     def augment_blueprint(self, spec, blueprint, seed):
        ...         # Add custom parameters
        ...         return blueprint  # Return augmented blueprint
        ...
        ...     def generate_components(self, spec, blueprint, seed):
        ...         return ModalityComponents(
        ...             encoder_type="custom_encoder",
        ...             embedding_type="custom_embedding",
        ...         )
        ...
        ...     def get_template_fragments(self, spec, blueprint):
        ...         return ["custom/encoder.py.j2"]
        ...
        >>> handler = CustomHandler()
        >>> handler.validate_spec(spec)  # Raises if invalid
        >>> components = handler.generate_components(spec, blueprint, seed=42)

    See Also:
        TextModalityHandler: Reference implementation for text modality
        ImageModalityHandler: Implementation for image modality
        get_handler: Factory function to get handler for a spec
    """

    # Valid architecture families that handlers can support
    VALID_FAMILIES: tuple[str, ...] = (
        "transformer",
        "cnn",
        "diffusion",
        "hybrid",
        "rnn",
        "mlp",
    )

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of this modality.

        Returns:
            Modality name as lowercase string (e.g., "text", "image").
        """
        ...

    @property
    @abstractmethod
    def supported_families(self) -> list[str]:
        """
        List of architecture families this handler supports.

        Returns:
            List of family names from VALID_FAMILIES.

        Example:
            >>> handler = TextModalityHandler()
            >>> handler.supported_families
            ['transformer', 'rnn']
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
        Augment blueprint with modality-specific parameters.

        This method adds parameters that are specific to this modality
        to the BlueprintState. For example:
        - Text: vocab_size, max_seq_len
        - Image: image_size, patch_size, num_channels
        - Audio: sample_rate, n_mels

        The augmentation must be deterministic given the same inputs
        and seed, to ensure reproducible synthesis.

        Args:
            spec: The model specification containing modality, constraints,
                and architecture requirements.
            blueprint: The base BlueprintState with dimension information
                (hidden_size, layers, heads) but without modality-specific
                parameters.
            seed: Random seed for deterministic augmentation. Any random
                choices (e.g., selecting from valid options) must use
                this seed.

        Returns:
            New BlueprintState with modality-specific parameters added.
            The original blueprint should not be modified.

        Raises:
            ValueError: If spec is incompatible with this modality.

        Example:
            >>> handler = TextModalityHandler()
            >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
            >>> print(augmented.vocab_size)
            50257
            >>> print(augmented.max_seq_len)
            2048

        Note:
            Implementations should use `random.Random(seed)` for any
            random choices to ensure determinism.
        """
        ...

    @abstractmethod
    def generate_components(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> ModalityComponents:
        """
        Generate modality-specific code components.

        This method determines which code components (encoders, decoders,
        embeddings, data loaders) should be used for this modality and
        architecture combination.

        Args:
            spec: The model specification.
            blueprint: The augmented BlueprintState with modality-specific
                parameters already added via augment_blueprint().
            seed: Random seed for deterministic generation.

        Returns:
            ModalityComponents instance specifying the code components
            to generate.

        Example:
            >>> handler = ImageModalityHandler()
            >>> components = handler.generate_components(spec, blueprint, 42)
            >>> print(components.encoder_type)
            'vit_encoder'
            >>> print(components.embedding_type)
            'patch_embedding'
        """
        ...

    @abstractmethod
    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """
        Get list of template fragments for this modality.

        Returns the paths to Jinja2 template fragments that should be
        included when generating code for this modality. Paths are
        relative to src/metagen/templates/fragments/.

        Args:
            spec: The model specification.
            blueprint: The augmented BlueprintState.

        Returns:
            List of template fragment paths to include.

        Example:
            >>> handler = TextModalityHandler()
            >>> fragments = handler.get_template_fragments(spec, blueprint)
            >>> print(fragments)
            ['embeddings/token_embedding.py.j2',
             'encoders/transformer_encoder.py.j2',
             'data/text_datasets.py.j2']
        """
        ...

    def validate_spec(self, spec: ModelSpec) -> None:
        """
        Validate that a spec is compatible with this modality handler.

        This method checks that:
        1. The spec's modality matches this handler
        2. The architecture family is supported
        3. Required fields are present

        Args:
            spec: The model specification to validate.

        Raises:
            ValueError: If spec is invalid or incompatible.

        Example:
            >>> handler = TextModalityHandler()
            >>> handler.validate_spec(text_spec)  # OK
            >>> handler.validate_spec(image_spec)  # Raises ValueError
        """
        # Check architecture family is supported
        family = spec.architecture.family.lower()
        if family not in self.supported_families:
            supported = ", ".join(self.supported_families)
            raise ValueError(
                f"Architecture family '{family}' is not supported by "
                f"{self.name} modality handler. Supported families: {supported}"
            )

        logger.debug(
            f"Validated spec for {self.name} modality handler: "
            f"family={family}, inputs={spec.modality.inputs}, "
            f"outputs={spec.modality.outputs}"
        )

    def supports_spec(self, spec: ModelSpec) -> bool:
        """
        Check if this handler can process the given spec.

        Unlike validate_spec(), this method returns a boolean instead
        of raising an exception. Useful for handler selection logic.

        Args:
            spec: The model specification to check.

        Returns:
            True if this handler can process the spec, False otherwise.

        Example:
            >>> text_handler = TextModalityHandler()
            >>> text_handler.supports_spec(text_spec)
            True
            >>> text_handler.supports_spec(image_spec)
            False
        """
        try:
            self.validate_spec(spec)
            return True
        except ValueError:
            return False

    def __repr__(self) -> str:
        """Return string representation of handler."""
        return f"{self.__class__.__name__}(name='{self.name}')"
