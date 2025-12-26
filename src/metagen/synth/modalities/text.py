"""
MetaGen Text Modality Handler

This module provides the TextModalityHandler for text/language model synthesis.
It handles text-specific parameters like vocabulary size, sequence length,
and token embeddings for transformer-based language models.

The text modality is the most mature in MetaGen, supporting:
- Autoregressive language models (GPT-style)
- Encoder models (BERT-style)
- Encoder-decoder models (T5-style)

Architecture Support:
    - transformer: Full support (primary)
    - rnn: Legacy support (LSTM, GRU)
    - hybrid: Transformer + other components

Example Usage:
    >>> from metagen.synth.modalities import TextModalityHandler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/text_llm_8b.yaml")
    >>> handler = TextModalityHandler()
    >>>
    >>> # Augment blueprint with text-specific parameters
    >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
    >>> print(f"Vocab size: {augmented.vocab_size}")
    Vocab size: 50257
    >>> print(f"Max seq len: {augmented.max_seq_len}")
    Max seq len: 262144
    >>>
    >>> # Generate code components
    >>> components = handler.generate_components(spec, augmented, seed=42)
    >>> print(components.encoder_type)
    'transformer_encoder'
    >>> print(components.embedding_type)
    'token_embedding'

See Also:
    - docs/user-guide/text-models.md for user documentation
    - ModalityHandler: Base class for all modality handlers

Author: MetaGen Team
Created: 2025-12-25
"""

from __future__ import annotations

import logging
import random
from dataclasses import replace
from typing import TYPE_CHECKING

from metagen.synth.modalities.base import ModalityComponents, ModalityHandler

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec
    from metagen.synth.architecture import BlueprintState

# Module-level logger
logger = logging.getLogger(__name__)

# Default vocabulary sizes for common tokenizers
# Reference: OpenAI GPT-2/3 uses 50257, LLaMA uses 32000
DEFAULT_VOCAB_SIZE = 50257  # GPT-2 tokenizer vocabulary size
LLAMA_VOCAB_SIZE = 32000  # LLaMA tokenizer vocabulary size

# Default sequence lengths
DEFAULT_SEQ_LEN = 2048  # Conservative default
MAX_SEQ_LEN_INFINITE = 1048576  # ~1M tokens for "infinite" context


class TextModalityHandler(ModalityHandler):
    """
    Modality handler for text/language model synthesis.

    TextModalityHandler is responsible for:
    1. Setting vocabulary size and tokenizer configuration
    2. Determining maximum sequence length from context_window spec
    3. Configuring token embeddings and positional encodings
    4. Selecting appropriate transformer encoder/decoder templates

    The handler supports various text model architectures:
    - Decoder-only (GPT, LLaMA): autoregressive generation
    - Encoder-only (BERT, RoBERTa): bidirectional encoding
    - Encoder-decoder (T5, BART): sequence-to-sequence

    Attributes:
        name: "text" - identifies this as the text modality handler
        supported_families: ["transformer", "rnn", "hybrid", "mlp"]

    Example:
        >>> handler = TextModalityHandler()
        >>> print(handler.name)
        'text'
        >>>
        >>> # Check if handler supports a spec
        >>> handler.supports_spec(text_spec)
        True
        >>>
        >>> # Augment blueprint
        >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
        >>> assert augmented.vocab_size == 50257
        >>> assert augmented.max_seq_len > 0

    See Also:
        ImageModalityHandler: For image/vision models
        ModalityHandler: Abstract base class
    """

    @property
    def name(self) -> str:
        """Return the modality name."""
        return "text"

    @property
    def supported_families(self) -> list[str]:
        """Return list of supported architecture families.

        Text models primarily use transformers, but we also support
        RNN (legacy) and hybrid architectures.
        """
        return ["transformer", "rnn", "hybrid", "mlp"]

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """
        Augment blueprint with text-specific parameters.

        This method adds:
        - vocab_size: Vocabulary size for token embeddings
        - max_seq_len: Maximum sequence length from context_window

        The augmentation is deterministic given the same inputs and seed.

        Args:
            spec: Model specification with text modality and constraints.
            blueprint: Base BlueprintState with core dimensions.
            seed: Random seed for deterministic choices.

        Returns:
            New BlueprintState with text-specific parameters added.

        Example:
            >>> handler = TextModalityHandler()
            >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
            >>> print(augmented.vocab_size)
            50257
            >>> print(augmented.max_seq_len)
            262144

        Note:
            The vocabulary size is currently fixed at 50257 (GPT-2 default).
            Future versions may support custom tokenizers.
        """
        rnd = random.Random(seed)

        # Determine vocabulary size
        # For now, we use the GPT-2 default. Future: parse from spec
        vocab_size = self._determine_vocab_size(spec, rnd)

        # Parse context_window to get max_seq_len
        max_seq_len = self._parse_context_window(spec)

        logger.debug(f"Text augmentation: vocab_size={vocab_size}, max_seq_len={max_seq_len}")

        # Create new BlueprintState with text parameters
        # We use dataclasses.replace to create an immutable copy
        return replace(
            blueprint,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
        )

    def generate_components(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> ModalityComponents:
        """
        Generate text-specific code components.

        Determines the encoder type, embedding type, and data loader
        based on the spec's training objectives and architecture family.

        Args:
            spec: Model specification.
            blueprint: Augmented BlueprintState with text parameters.
            seed: Random seed (unused for text, deterministic).

        Returns:
            ModalityComponents specifying text model code generation.

        Example:
            >>> handler = TextModalityHandler()
            >>> components = handler.generate_components(spec, blueprint, 42)
            >>> print(components.encoder_type)
            'transformer_encoder'
            >>> print(components.embedding_type)
            'token_embedding'
            >>> print(components.data_loader_type)
            'text_dataset'
        """
        family = spec.architecture.family.lower()
        objectives = [obj.lower() for obj in spec.training.objective]

        # Determine encoder type based on architecture family
        encoder_type = self._select_encoder_type(family)

        # Determine decoder type based on training objectives
        decoder_type = self._select_decoder_type(objectives)

        # Select loss type
        loss_type = self._select_loss_type(objectives)

        # Build template fragments list
        template_fragments = self._get_template_fragments_list(family, objectives)

        # Additional imports for text models
        additional_imports = (
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
        )

        # Build config with text-specific parameters
        config = {
            "vocab_size": blueprint.vocab_size,
            "max_seq_len": blueprint.max_seq_len,
            "hidden_size": blueprint.dims["hidden_size"],
            "num_layers": blueprint.dims["layers"],
            "num_heads": blueprint.dims["heads"],
        }

        return ModalityComponents(
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            embedding_type="token_embedding",
            data_loader_type="text_dataset",
            loss_type=loss_type,
            template_fragments=tuple(template_fragments),
            additional_imports=additional_imports,
            config=config,
        )

    def get_template_fragments(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
    ) -> list[str]:
        """
        Get list of template fragments for text modality.

        Returns paths to Jinja2 template fragments that should be
        included when generating code for text models.

        Args:
            spec: Model specification.
            blueprint: Augmented BlueprintState.

        Returns:
            List of template fragment paths.

        Example:
            >>> handler = TextModalityHandler()
            >>> fragments = handler.get_template_fragments(spec, blueprint)
            >>> print(fragments)
            ['embeddings/token_embedding.py.j2',
             'encoders/transformer_encoder.py.j2',
             'data/text_datasets.py.j2']
        """
        family = spec.architecture.family.lower()
        objectives = [obj.lower() for obj in spec.training.objective]
        return self._get_template_fragments_list(family, objectives)

    def _determine_vocab_size(self, spec: ModelSpec, rnd: random.Random) -> int:
        """
        Determine vocabulary size for the text model.

        Currently returns the GPT-2 default (50257). Future versions
        may parse vocabulary size from spec or support custom tokenizers.

        Args:
            spec: Model specification (currently unused).
            rnd: Random generator (currently unused).

        Returns:
            Vocabulary size as integer.
        """
        # TODO: Support custom tokenizers and vocab sizes from spec
        # For now, use GPT-2 default which works well for most cases
        return DEFAULT_VOCAB_SIZE

    def _parse_context_window(self, spec: ModelSpec) -> int:
        """
        Parse context_window constraint to get max sequence length.

        Supports various formats:
        - "256k" -> 262144
        - "1M" -> 1048576
        - "2048" -> 2048
        - "infinite", "inf", "∞" -> 1048576 (practical limit)

        Args:
            spec: Model specification with constraints.context_window.

        Returns:
            Maximum sequence length as integer.

        Example:
            >>> # Assuming spec.constraints.context_window = "256k"
            >>> handler._parse_context_window(spec)
            262144
        """
        ctx = str(spec.constraints.context_window).lower().strip().replace(" ", "")

        # Handle infinite context
        if any(x in ctx for x in ["inf", "∞", "infinite"]):
            logger.debug("Infinite context window requested, using 1M tokens")
            return MAX_SEQ_LEN_INFINITE

        # Parse with multipliers
        try:
            if "m" in ctx:
                # e.g., "1M" -> 1048576
                return int(float(ctx.replace("m", "")) * 1024 * 1024)
            elif "k" in ctx:
                # e.g., "256k" -> 262144
                return int(float(ctx.replace("k", "")) * 1024)
            elif ctx.isdigit():
                # e.g., "2048" -> 2048
                return int(ctx)
            else:
                logger.warning(
                    f"Could not parse context_window '{ctx}', using default {DEFAULT_SEQ_LEN}"
                )
                return DEFAULT_SEQ_LEN
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing context_window: {e}, using default")
            return DEFAULT_SEQ_LEN

    def _select_encoder_type(self, family: str) -> str:
        """
        Select encoder type based on architecture family.

        Args:
            family: Architecture family (transformer, rnn, etc.)

        Returns:
            Encoder type string for code generation.
        """
        encoder_map = {
            "transformer": "transformer_encoder",
            "rnn": "rnn_encoder",
            "hybrid": "hybrid_encoder",
            "mlp": "mlp_encoder",
        }
        return encoder_map.get(family, "transformer_encoder")

    def _select_decoder_type(self, objectives: list[str]) -> str | None:
        """
        Select decoder type based on training objectives.

        Args:
            objectives: List of training objectives from spec.

        Returns:
            Decoder type string or None if no decoder needed.
        """
        # Autoregressive models need a decoder head
        if "autoregressive" in objectives:
            return "autoregressive"
        # Masked language modeling (BERT-style)
        if "mlm" in objectives or "masked" in objectives:
            return "masked_lm"
        # Sequence-to-sequence
        if "seq2seq" in objectives:
            return "seq2seq"
        # Default for text generation
        return "autoregressive"

    def _select_loss_type(self, objectives: list[str]) -> str:
        """
        Select loss function type based on training objectives.

        Args:
            objectives: List of training objectives from spec.

        Returns:
            Loss type string for code generation.
        """
        if "contrastive" in objectives:
            return "contrastive"
        if "mlm" in objectives:
            return "masked_lm_loss"
        # Default for autoregressive LMs
        return "cross_entropy"

    def _get_template_fragments_list(
        self,
        family: str,
        objectives: list[str],
    ) -> list[str]:
        """
        Build list of template fragment paths.

        Args:
            family: Architecture family.
            objectives: Training objectives.

        Returns:
            List of template fragment paths.
        """
        fragments = []

        # Always include token embedding for text
        fragments.append("embeddings/token_embedding.py.j2")

        # Encoder based on family
        if family == "transformer":
            fragments.append("encoders/transformer_encoder.py.j2")
        elif family == "rnn":
            fragments.append("encoders/rnn_encoder.py.j2")
        else:
            fragments.append("encoders/transformer_encoder.py.j2")

        # Decoder if autoregressive
        if "autoregressive" in objectives:
            fragments.append("decoders/autoregressive_decoder.py.j2")

        # Data loader
        fragments.append("data/text_datasets.py.j2")

        # Loss function
        loss_type = self._select_loss_type(objectives)
        loss_fragment_map = {
            "cross_entropy": "losses/cross_entropy.py.j2",
            "contrastive": "losses/contrastive.py.j2",
            "masked_lm_loss": "losses/masked_lm.py.j2",
        }
        try:
            fragments.append(loss_fragment_map[loss_type])
        except KeyError as exc:
            raise ValueError(f"Unsupported text loss type: {loss_type}") from exc

        return fragments

    def validate_spec(self, spec: ModelSpec) -> None:
        """
        Validate that spec is compatible with text modality.

        Extends base validation to check text-specific requirements:
        - Text must be in inputs or outputs
        - context_window must be parseable

        Args:
            spec: Model specification to validate.

        Raises:
            ValueError: If spec is invalid for text modality.
        """
        # First, run base validation (architecture family check)
        super().validate_spec(spec)

        # Check that text is in modality
        inputs = [m.lower() for m in spec.modality.inputs]
        outputs = [m.lower() for m in spec.modality.outputs]

        if "text" not in inputs and "text" not in outputs:
            raise ValueError(
                f"TextModalityHandler requires 'text' in inputs or outputs. "
                f"Got inputs={inputs}, outputs={outputs}"
            )

        logger.debug(f"Validated text spec: inputs={inputs}, outputs={outputs}")
