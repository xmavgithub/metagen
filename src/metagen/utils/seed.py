import hashlib
import random


def derive_seed(spec_content: str, base_seed: int = 42) -> int:
    """Create a deterministic seed from spec content."""
    digest = hashlib.sha256(spec_content.encode("utf-8")).digest()
    derived = int.from_bytes(digest[:4], "big")
    return (base_seed + derived) % (2**31 - 1)


def set_global_seed(seed: int | None) -> int:
    """Seed random modules deterministically."""
    resolved = seed if seed is not None else 42
    random.seed(resolved)
    return resolved
