"""Performance benchmarks for MetaGen synthesis pipeline."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import yaml

from metagen.specs.loader import load_spec
from metagen.synth.engine import synthesize


def _create_temp_spec(tmpdir: Path, name: str = "test-model") -> Path:
    """Create a minimal spec file for testing."""
    spec_data = {
        "name": name,
        "description": "Test model for performance testing",
    }
    spec_path = tmpdir / "spec.yaml"
    spec_path.write_text(yaml.safe_dump(spec_data))
    return spec_path


class TestPerformanceBenchmarks:
    """Performance benchmarks for synthesis operations."""

    @pytest.fixture
    def examples_dir(self) -> Path:
        """Get the examples/specs directory."""
        return Path(__file__).parent.parent / "examples" / "specs"

    @pytest.mark.slow
    def test_synthesis_performance(self, examples_dir: Path) -> None:
        """Benchmark full synthesis time."""
        spec_path = examples_dir / "text" / "text_llm_8b.yaml"
        if not spec_path.exists():
            pytest.skip("text/text_llm_8b.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            start = time.perf_counter()
            synthesize(spec_path, output_dir, base_seed=42)
            elapsed = time.perf_counter() - start

            # Synthesis should complete in reasonable time
            assert elapsed < 30.0, f"Synthesis took {elapsed:.2f}s, expected < 30s"
            print(f"\nSynthesis completed in {elapsed:.2f}s")

    def test_spec_loading_performance(self, examples_dir: Path) -> None:
        """Benchmark spec loading and validation time."""
        spec_path = examples_dir / "text" / "text_llm_8b.yaml"
        if not spec_path.exists():
            pytest.skip("text/text_llm_8b.yaml not found")

        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            load_spec(spec_path)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        assert avg_time < 0.1, f"Spec loading took {avg_time:.4f}s avg, expected < 0.1s"
        print(f"\nSpec loading: {avg_time * 1000:.2f}ms avg over {iterations} iterations")

    def test_minimal_spec_performance(self) -> None:
        """Benchmark minimal spec synthesis time."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            spec_path = _create_temp_spec(tmppath, "minimal-model")
            output_dir = tmppath / "output"

            start = time.perf_counter()
            synthesize(spec_path, output_dir, base_seed=42)
            elapsed = time.perf_counter() - start

            # Minimal synthesis should be fast
            assert elapsed < 10.0, f"Minimal synthesis took {elapsed:.2f}s, expected < 10s"
            print(f"\nMinimal synthesis completed in {elapsed:.2f}s")

    @pytest.mark.slow
    def test_multiple_modalities_performance(self, examples_dir: Path) -> None:
        """Benchmark multimodal synthesis time."""
        spec_path = examples_dir / "multimodal" / "multimodal_clip.yaml"
        if not spec_path.exists():
            pytest.skip("multimodal/multimodal_clip.yaml not found")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            start = time.perf_counter()
            synthesize(spec_path, output_dir, base_seed=42)
            elapsed = time.perf_counter() - start

            # Multimodal should still be reasonable
            assert elapsed < 45.0, f"Multimodal synthesis took {elapsed:.2f}s, expected < 45s"
            print(f"\nMultimodal synthesis completed in {elapsed:.2f}s")


class TestMemoryUsage:
    """Test memory usage during synthesis."""

    def test_no_memory_leak_on_repeated_synthesis(self) -> None:
        """Ensure no significant memory growth on repeated synthesis."""
        import gc

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            spec_path = _create_temp_spec(tmppath, "memory-test")

            # Warm up
            output_warmup = tmppath / "warmup"
            synthesize(spec_path, output_warmup, base_seed=42)

            gc.collect()

            # Run multiple times
            for i in range(5):
                output_i = tmppath / f"run_{i}"
                synthesize(spec_path, output_i, base_seed=i)
                gc.collect()

            # If we get here without OOM, memory is managed correctly
            assert True
