"""Caching utilities for MetaGen synthesis pipeline."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any


class SynthesisCache:
    """Simple file-based cache for synthesis results.

    Caches intermediate results based on spec hash to avoid
    redundant computation during architecture search.
    """

    def __init__(self, cache_dir: Path | None = None, enabled: bool = True) -> None:
        """Initialize cache.

        Args:
            cache_dir: Directory for cache files. If None, caching is memory-only.
            enabled: Whether caching is enabled.
        """
        self.cache_dir = cache_dir
        self.enabled = enabled
        self._memory_cache: dict[str, Any] = {}

        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _compute_key(self, spec_dict: dict, seed: int) -> str:
        """Compute cache key from spec and seed."""
        content = json.dumps(spec_dict, sort_keys=True) + str(seed)
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, spec_dict: dict, seed: int) -> Any | None:
        """Get cached result if available."""
        if not self.enabled:
            return None

        key = self._compute_key(spec_dict, seed)

        # Check memory cache first
        if key in self._memory_cache:
            return self._memory_cache[key]

        # Check file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            if cache_file.exists():
                try:
                    data = json.loads(cache_file.read_text())
                    self._memory_cache[key] = data
                    return data
                except (json.JSONDecodeError, OSError):
                    pass

        return None

    def set(self, spec_dict: dict, seed: int, value: Any) -> None:
        """Cache a result."""
        if not self.enabled:
            return

        key = self._compute_key(spec_dict, seed)
        self._memory_cache[key] = value

        # Write to file cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{key}.json"
            try:
                cache_file.write_text(json.dumps(value))
            except (TypeError, OSError):
                pass  # Skip non-serializable or write errors

    def clear(self) -> None:
        """Clear all cached data."""
        self._memory_cache.clear()
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()


# Global cache instance (disabled by default for determinism)
_global_cache: SynthesisCache | None = None


def get_cache() -> SynthesisCache:
    """Get or create global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = SynthesisCache(enabled=False)
    return _global_cache


def enable_cache(cache_dir: Path | None = None) -> SynthesisCache:
    """Enable global caching.

    Args:
        cache_dir: Optional directory for persistent cache.

    Returns:
        The cache instance.
    """
    global _global_cache
    _global_cache = SynthesisCache(cache_dir=cache_dir, enabled=True)
    return _global_cache


def disable_cache() -> None:
    """Disable global caching."""
    global _global_cache
    if _global_cache:
        _global_cache.clear()
    _global_cache = SynthesisCache(enabled=False)


@lru_cache(maxsize=128)
def cached_param_estimate(
    d_model: int,
    n_layers: int,
    n_heads: int,
    vocab_size: int,
    intermediate_factor: float = 4.0,
) -> int:
    """Cached parameter count estimation for transformer architectures.

    Args:
        d_model: Model dimension.
        n_layers: Number of layers.
        n_heads: Number of attention heads.
        vocab_size: Vocabulary size.
        intermediate_factor: FFN intermediate size factor.

    Returns:
        Estimated parameter count.
    """
    # Embedding
    embed_params = vocab_size * d_model

    # Per layer
    attn_params = 4 * d_model * d_model  # Q, K, V, O projections
    ff_params = 2 * d_model * int(d_model * intermediate_factor)  # up + down
    layer_norm_params = 4 * d_model  # 2 layer norms per layer
    layer_params = attn_params + ff_params + layer_norm_params

    # Final layer norm + output projection
    output_params = d_model + vocab_size * d_model

    return embed_params + (n_layers * layer_params) + output_params
