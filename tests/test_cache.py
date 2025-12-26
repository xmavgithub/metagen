"""Tests for caching utilities."""

from __future__ import annotations

import tempfile
from pathlib import Path

from metagen.utils.cache import (
    SynthesisCache,
    cached_param_estimate,
    disable_cache,
    enable_cache,
    get_cache,
)


class TestSynthesisCache:
    """Tests for SynthesisCache class."""

    def test_memory_cache(self) -> None:
        """Test in-memory caching."""
        cache = SynthesisCache(enabled=True)
        spec = {"name": "test", "layers": 12}

        # Initially empty
        assert cache.get(spec, seed=42) is None

        # Set and get
        cache.set(spec, 42, {"result": "cached"})
        assert cache.get(spec, 42) == {"result": "cached"}

        # Different seed = different key
        assert cache.get(spec, seed=99) is None

    def test_file_cache(self) -> None:
        """Test file-based caching."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = SynthesisCache(cache_dir=cache_dir, enabled=True)

            spec = {"name": "test"}
            cache.set(spec, 42, {"data": [1, 2, 3]})

            # Verify file was created
            assert len(list(cache_dir.glob("*.json"))) == 1

            # Create new cache instance to test persistence
            cache2 = SynthesisCache(cache_dir=cache_dir, enabled=True)
            assert cache2.get(spec, seed=42) == {"data": [1, 2, 3]}

    def test_disabled_cache(self) -> None:
        """Test that disabled cache doesn't store anything."""
        cache = SynthesisCache(enabled=False)
        spec = {"name": "test"}

        cache.set(spec, 42, {"result": "data"})
        assert cache.get(spec, 42) is None

    def test_clear_cache(self) -> None:
        """Test cache clearing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = SynthesisCache(cache_dir=cache_dir, enabled=True)

            spec = {"name": "test"}
            cache.set(spec, 42, {"data": "value"})

            cache.clear()

            assert cache.get(spec, 42) is None
            assert len(list(cache_dir.glob("*.json"))) == 0


class TestGlobalCache:
    """Tests for global cache functions."""

    def test_get_cache_default_disabled(self) -> None:
        """Test that global cache is disabled by default."""
        disable_cache()  # Reset state
        cache = get_cache()
        assert not cache.enabled

    def test_enable_disable_cache(self) -> None:
        """Test enabling and disabling global cache."""
        disable_cache()

        cache = enable_cache()
        assert cache.enabled

        disable_cache()
        cache = get_cache()
        assert not cache.enabled

    def test_enable_with_directory(self) -> None:
        """Test enabling cache with file directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "cache"
            cache = enable_cache(cache_dir)

            assert cache.enabled
            assert cache.cache_dir == cache_dir
            assert cache_dir.exists()

            disable_cache()


class TestCachedParamEstimate:
    """Tests for cached parameter estimation."""

    def test_basic_estimate(self) -> None:
        """Test parameter count estimation."""
        params = cached_param_estimate(
            d_model=768,
            n_layers=12,
            n_heads=12,
            vocab_size=50000,
        )

        # Should be in reasonable range for a 110M-ish model
        assert 80_000_000 < params < 200_000_000

    def test_caching_works(self) -> None:
        """Test that caching is effective (same call returns same result)."""
        args = (512, 6, 8, 30000)

        result1 = cached_param_estimate(*args)
        result2 = cached_param_estimate(*args)

        assert result1 == result2

    def test_different_configs(self) -> None:
        """Test different configurations produce different results."""
        small = cached_param_estimate(256, 4, 4, 10000)
        large = cached_param_estimate(1024, 24, 16, 100000)

        assert large > small
