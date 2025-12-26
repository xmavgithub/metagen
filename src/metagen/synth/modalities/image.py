"""
MetaGen Image Modality Handler

This module provides the ImageModalityHandler for image/vision model synthesis.
It handles image-specific parameters like image size, patch size, number of
channels, and supports various vision architectures.

The image modality supports:
- Vision Transformers (ViT): Patch-based transformer encoders
- Convolutional Networks (CNN): ResNet-style architectures
- Diffusion Models: U-Net based architectures for generation
- Hybrid Models: CNN + Transformer combinations

Architecture Support:
    - transformer: ViT (Vision Transformer)
    - cnn: ResNet, EfficientNet-style
    - diffusion: U-Net for diffusion models
    - hybrid: CNN backbone + Transformer head

Example Usage:
    >>> from metagen.synth.modalities import ImageModalityHandler
    >>> from metagen.specs.loader import load_spec
    >>>
    >>> spec = load_spec("examples/specs/image_vit_base.yaml")
    >>> handler = ImageModalityHandler()
    >>>
    >>> # Augment blueprint with image-specific parameters
    >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
    >>> print(f"Image size: {augmented.image_size}")
    Image size: 224
    >>> print(f"Patch size: {augmented.patch_size}")
    Patch size: 16
    >>> print(f"Num patches: {augmented.num_patches}")
    Num patches: 196
    >>>
    >>> # Generate code components
    >>> components = handler.generate_components(spec, augmented, seed=42)
    >>> print(components.encoder_type)
    'vit_encoder'
    >>> print(components.embedding_type)
    'patch_embedding'

See Also:
    - docs/user-guide/image-models.md for user documentation
    - TextModalityHandler: Reference implementation for text modality
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

# Default image sizes for common vision tasks
# Reference: ImageNet uses 224, CLIP uses 224/336, larger models use 384/512
DEFAULT_IMAGE_SIZE = 224
IMAGENET_SIZE = 224
CLIP_SIZE = 224
LARGE_MODEL_SIZE = 384

# Default patch sizes for ViT
# Reference: ViT-Base uses 16, ViT-Large uses 14, ViT-Huge uses 14
DEFAULT_PATCH_SIZE = 16
VIT_BASE_PATCH = 16
VIT_LARGE_PATCH = 14

# Default number of channels
RGB_CHANNELS = 3
GRAYSCALE_CHANNELS = 1
RGBA_CHANNELS = 4

# Common image size presets (width x height)
IMAGE_SIZE_PRESETS = {
    "imagenet": 224,
    "imagenet_large": 384,
    "clip": 224,
    "clip_large": 336,
    "sd": 512,  # Stable Diffusion
    "sd_xl": 1024,  # SDXL
    "hd": 1080,
    "4k": 2160,
}

# Patch size presets
PATCH_SIZE_PRESETS = {
    "vit_base": 16,
    "vit_large": 14,
    "vit_huge": 14,
    "small": 8,
    "default": 16,
}


class ImageModalityHandler(ModalityHandler):
    """
    Modality handler for image/vision model synthesis.

    ImageModalityHandler is responsible for:
    1. Setting image dimensions (image_size, num_channels)
    2. Computing patch configuration for ViT models
    3. Configuring encoder architecture (ViT, CNN, U-Net)
    4. Selecting appropriate template fragments

    The handler supports various vision model architectures:
    - Vision Transformers (ViT): Patch-based attention models
    - CNNs: Convolutional networks for classification/detection
    - Diffusion: U-Net based generative models
    - Hybrid: CNN encoders with transformer heads

    Attributes:
        name: "image" - identifies this as the image modality handler
        supported_families: ["transformer", "cnn", "diffusion", "hybrid"]

    Example:
        >>> handler = ImageModalityHandler()
        >>> print(handler.name)
        'image'
        >>>
        >>> # Check if handler supports a spec
        >>> handler.supports_spec(vit_spec)
        True
        >>>
        >>> # Augment blueprint
        >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
        >>> assert augmented.image_size == 224
        >>> assert augmented.patch_size == 16
        >>> assert augmented.num_patches == 196  # (224/16)^2

    See Also:
        TextModalityHandler: For text/language models
        ModalityHandler: Abstract base class
    """

    @property
    def name(self) -> str:
        """Return the modality name."""
        return "image"

    @property
    def supported_families(self) -> list[str]:
        """Return list of supported architecture families.

        Image models support multiple architecture families:
        - transformer: Vision Transformers (ViT)
        - cnn: Convolutional Neural Networks
        - diffusion: U-Net based diffusion models
        - hybrid: Combined architectures
        """
        return ["transformer", "cnn", "diffusion", "hybrid"]

    def augment_blueprint(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> BlueprintState:
        """
        Augment blueprint with image-specific parameters.

        This method adds:
        - image_size: Image resolution (square images assumed)
        - num_channels: Number of color channels (3 for RGB)
        - patch_size: Patch size for ViT models
        - num_patches: Computed number of patches

        The augmentation is deterministic given the same inputs and seed.

        Args:
            spec: Model specification with image modality and constraints.
            blueprint: Base BlueprintState with core dimensions.
            seed: Random seed for deterministic choices.

        Returns:
            New BlueprintState with image-specific parameters added.

        Example:
            >>> handler = ImageModalityHandler()
            >>> augmented = handler.augment_blueprint(spec, blueprint, seed=42)
            >>> print(augmented.image_size)
            224
            >>> print(augmented.patch_size)
            16
            >>> print(augmented.num_patches)
            196

        Note:
            For ViT models, num_patches = (image_size / patch_size)^2
            This must be compatible with the hidden_size for position embeddings.
        """
        rnd = random.Random(seed)

        # Determine image size from spec or use default
        image_size = self._determine_image_size(spec, rnd)

        # Determine number of channels
        num_channels = self._determine_num_channels(spec)

        # Determine patch size (for ViT architectures)
        patch_size = self._determine_patch_size(spec, image_size, rnd)

        # Compute number of patches
        num_patches = self._compute_num_patches(image_size, patch_size)

        logger.debug(
            f"Image augmentation: image_size={image_size}, "
            f"num_channels={num_channels}, patch_size={patch_size}, "
            f"num_patches={num_patches}"
        )

        # Create new BlueprintState with image parameters
        return replace(
            blueprint,
            image_size=image_size,
            num_channels=num_channels,
            patch_size=patch_size,
            num_patches=num_patches,
        )

    def generate_components(
        self,
        spec: ModelSpec,
        blueprint: BlueprintState,
        seed: int,
    ) -> ModalityComponents:
        """
        Generate image-specific code components.

        Determines the encoder type, embedding type, and data loader
        based on the spec's architecture family and training objectives.

        Args:
            spec: Model specification.
            blueprint: Augmented BlueprintState with image parameters.
            seed: Random seed (unused for image, deterministic).

        Returns:
            ModalityComponents specifying image model code generation.

        Example:
            >>> handler = ImageModalityHandler()
            >>> components = handler.generate_components(spec, blueprint, 42)
            >>> print(components.encoder_type)
            'vit_encoder'
            >>> print(components.embedding_type)
            'patch_embedding'
            >>> print(components.data_loader_type)
            'image_folder'
        """
        family = spec.architecture.family.lower()
        objectives = [obj.lower() for obj in spec.training.objective]

        # Determine encoder type based on architecture family
        encoder_type = self._select_encoder_type(family)

        # Determine embedding type
        embedding_type = self._select_embedding_type(family)

        # Determine decoder type based on objectives
        decoder_type = self._select_decoder_type(family, objectives)

        # Select loss type
        loss_type = self._select_loss_type(objectives)

        # Build template fragments list
        template_fragments = self._get_template_fragments_list(family, objectives)

        # Additional imports for image models
        additional_imports = (
            "import torch",
            "import torch.nn as nn",
            "import torch.nn.functional as F",
            "from torchvision import transforms",
        )

        # Build config with image-specific parameters
        config = {
            "image_size": blueprint.image_size,
            "num_channels": blueprint.num_channels,
            "patch_size": blueprint.patch_size,
            "num_patches": blueprint.num_patches,
            "hidden_size": blueprint.dims["hidden_size"],
            "num_layers": blueprint.dims["layers"],
            "num_heads": blueprint.dims["heads"],
        }

        return ModalityComponents(
            encoder_type=encoder_type,
            decoder_type=decoder_type,
            embedding_type=embedding_type,
            data_loader_type="image_folder",
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
        Get list of template fragments for image modality.

        Returns paths to Jinja2 template fragments that should be
        included when generating code for image models.

        Args:
            spec: Model specification.
            blueprint: Augmented BlueprintState.

        Returns:
            List of template fragment paths.

        Example:
            >>> handler = ImageModalityHandler()
            >>> fragments = handler.get_template_fragments(spec, blueprint)
            >>> print(fragments)
            ['embeddings/patch_embedding.py.j2',
             'encoders/vit_encoder.py.j2',
             'data/image_datasets.py.j2']
        """
        family = spec.architecture.family.lower()
        objectives = [obj.lower() for obj in spec.training.objective]
        return self._get_template_fragments_list(family, objectives)

    def _determine_image_size(self, spec: ModelSpec, rnd: random.Random) -> int:
        """
        Determine image size for the model.

        Parses image_size from spec constraints if available,
        otherwise uses intelligent defaults based on task.

        Args:
            spec: Model specification.
            rnd: Random generator for choices.

        Returns:
            Image size as integer (square images assumed).
        """
        # Check if image_size is specified in constraints
        if hasattr(spec.constraints, "image_size"):
            size_spec = str(spec.constraints.image_size).lower().strip()

            # Check presets
            if size_spec in IMAGE_SIZE_PRESETS:
                return IMAGE_SIZE_PRESETS[size_spec]

            # Parse numeric value
            try:
                return int(size_spec)
            except (ValueError, TypeError):
                pass

        # Check task type for intelligent defaults
        task_type = spec.task.type.lower() if hasattr(spec.task, "type") else ""

        if "generation" in task_type or "diffusion" in task_type:
            return 512  # Larger for generative models
        elif "classification" in task_type:
            return IMAGENET_SIZE

        # Default for vision models
        return DEFAULT_IMAGE_SIZE

    def _determine_num_channels(self, spec: ModelSpec) -> int:
        """
        Determine number of image channels.

        Args:
            spec: Model specification.

        Returns:
            Number of channels (typically 3 for RGB).
        """
        # Check if channels specified in constraints
        if hasattr(spec.constraints, "num_channels"):
            try:
                return int(spec.constraints.num_channels)
            except (ValueError, TypeError):
                pass

        # Check for grayscale hint
        if hasattr(spec.task, "domain"):
            domain = spec.task.domain.lower()
            if "grayscale" in domain or "medical" in domain:
                return GRAYSCALE_CHANNELS

        # Default to RGB
        return RGB_CHANNELS

    def _determine_patch_size(
        self,
        spec: ModelSpec,
        image_size: int,
        rnd: random.Random,
    ) -> int:
        """
        Determine patch size for ViT models.

        Args:
            spec: Model specification.
            image_size: Image size to ensure compatibility.
            rnd: Random generator.

        Returns:
            Patch size as integer.
        """
        family = spec.architecture.family.lower()

        # Non-transformer models don't use patches
        if family not in ["transformer", "hybrid"]:
            return 1  # Pixel-level for CNNs

        # Check if patch_size specified
        if hasattr(spec.constraints, "patch_size"):
            try:
                patch_size = int(spec.constraints.patch_size)
                if image_size % patch_size == 0:
                    return patch_size
            except (ValueError, TypeError):
                pass

        # Choose based on parameter budget
        if hasattr(spec.constraints, "parameter_budget"):
            budget = spec.constraints.parameter_budget
            if hasattr(budget, "max"):
                max_params = self._parse_param_budget(str(budget.max))
                if max_params and max_params > 1_000_000_000:  # > 1B
                    return VIT_LARGE_PATCH
                elif max_params and max_params > 100_000_000:  # > 100M
                    return VIT_BASE_PATCH

        # Default: ensure divisibility
        for patch in [16, 14, 8, 32]:
            if image_size % patch == 0:
                return patch

        return DEFAULT_PATCH_SIZE

    def _parse_param_budget(self, budget_str: str) -> int | None:
        """Parse parameter budget string like '1B', '100M'."""
        budget_str = budget_str.lower().strip()
        try:
            if "b" in budget_str:
                return int(float(budget_str.replace("b", "")) * 1_000_000_000)
            elif "m" in budget_str:
                return int(float(budget_str.replace("m", "")) * 1_000_000)
            elif "k" in budget_str:
                return int(float(budget_str.replace("k", "")) * 1_000)
            else:
                return int(budget_str)
        except (ValueError, TypeError):
            return None

    def _compute_num_patches(self, image_size: int, patch_size: int) -> int:
        """
        Compute number of patches for ViT.

        Args:
            image_size: Image dimension.
            patch_size: Patch dimension.

        Returns:
            Number of patches (image_size / patch_size)^2
        """
        if patch_size <= 0:
            return 1

        patches_per_side = image_size // patch_size
        return patches_per_side * patches_per_side

    def _select_encoder_type(self, family: str) -> str:
        """
        Select encoder type based on architecture family.

        Args:
            family: Architecture family (transformer, cnn, etc.)

        Returns:
            Encoder type string for code generation.
        """
        encoder_map = {
            "transformer": "vit_encoder",
            "cnn": "resnet_encoder",
            "diffusion": "unet_encoder",
            "hybrid": "hybrid_vit_cnn_encoder",
        }
        return encoder_map.get(family, "vit_encoder")

    def _select_embedding_type(self, family: str) -> str:
        """
        Select embedding type based on architecture family.

        Args:
            family: Architecture family.

        Returns:
            Embedding type string.
        """
        embedding_map = {
            "transformer": "patch_embedding",
            "cnn": "conv_stem",
            "diffusion": "conv_embedding",
            "hybrid": "patch_embedding",
        }
        return embedding_map.get(family, "patch_embedding")

    def _select_decoder_type(self, family: str, objectives: list[str]) -> str | None:
        """
        Select decoder type based on family and objectives.

        Args:
            family: Architecture family.
            objectives: Training objectives.

        Returns:
            Decoder type string or None.
        """
        # Diffusion models always need decoder
        if family == "diffusion" or "diffusion" in objectives:
            return "diffusion_decoder"

        # Generation tasks need decoders
        if any(obj in objectives for obj in ["generation", "reconstruction", "vae"]):
            return "image_decoder"

        # Classification doesn't need decoder
        if "classification" in objectives or "contrastive" in objectives:
            return "classification_head"

        return None

    def _select_loss_type(self, objectives: list[str]) -> str:
        """
        Select loss function type based on training objectives.

        Args:
            objectives: List of training objectives.

        Returns:
            Loss type string.
        """
        if "diffusion" in objectives:
            return "diffusion_loss"
        if "contrastive" in objectives:
            return "contrastive_loss"
        if "reconstruction" in objectives or "vae" in objectives:
            return "reconstruction_loss"
        if "classification" in objectives:
            return "cross_entropy"

        # Default for image models
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

        # Embedding based on family
        if family == "transformer" or family == "hybrid":
            fragments.append("embeddings/patch_embedding.py.j2")
        else:
            fragments.append("embeddings/conv_embedding.py.j2")

        # Encoder based on family
        if family == "transformer":
            fragments.append("encoders/vit_encoder.py.j2")
        elif family == "cnn":
            fragments.append("encoders/resnet_encoder.py.j2")
        elif family == "diffusion":
            fragments.append("encoders/unet_encoder.py.j2")
        elif family == "hybrid":
            fragments.append("encoders/hybrid_encoder.py.j2")
        else:
            fragments.append("encoders/vit_encoder.py.j2")

        # Decoder if needed
        if family == "diffusion" or "diffusion" in objectives:
            fragments.append("decoders/diffusion_decoder.py.j2")
        elif "generation" in objectives or "vae" in objectives:
            fragments.append("decoders/image_decoder.py.j2")

        # Data loader
        fragments.append("data/image_datasets.py.j2")

        # Loss function
        if "diffusion" in objectives:
            fragments.append("losses/diffusion_loss.py.j2")
        elif "contrastive" in objectives:
            fragments.append("losses/contrastive_loss.py.j2")
        else:
            fragments.append("losses/cross_entropy.py.j2")

        return fragments

    def validate_spec(self, spec: ModelSpec) -> None:
        """
        Validate that spec is compatible with image modality.

        Extends base validation to check image-specific requirements:
        - Image must be in inputs or outputs
        - Image size must be valid if specified

        Args:
            spec: Model specification to validate.

        Raises:
            ValueError: If spec is invalid for image modality.
        """
        # First, run base validation (architecture family check)
        super().validate_spec(spec)

        # Check that image is in modality
        inputs = [m.lower() for m in spec.modality.inputs]
        outputs = [m.lower() for m in spec.modality.outputs]

        if "image" not in inputs and "image" not in outputs:
            raise ValueError(
                f"ImageModalityHandler requires 'image' in inputs or outputs. "
                f"Got inputs={inputs}, outputs={outputs}"
            )

        # Validate image_size if specified
        if hasattr(spec.constraints, "image_size"):
            size = spec.constraints.image_size
            if isinstance(size, int) and size <= 0:
                raise ValueError(f"image_size must be positive, got {size}")

        logger.debug(f"Validated image spec: inputs={inputs}, outputs={outputs}")
