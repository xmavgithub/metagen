"""
MetaGen Paper Templates

LaTeX templates for generating publication-quality academic papers.
Provides professional formatting combining NeurIPS clarity with ICLR typography.
"""

from __future__ import annotations

from pathlib import Path

# Template directory
TEMPLATES_DIR = Path(__file__).parent


def get_preamble() -> str:
    """Get the LaTeX preamble content."""
    preamble_path = TEMPLATES_DIR / "preamble.tex"
    return preamble_path.read_text(encoding="utf-8")


def get_style() -> str:
    """Get the custom style file content."""
    style_path = TEMPLATES_DIR / "metagen_academic.sty"
    return style_path.read_text(encoding="utf-8")


def get_main_template() -> str:
    """Get the main document template."""
    template_path = TEMPLATES_DIR / "main_template.tex"
    return template_path.read_text(encoding="utf-8")


def copy_templates_to(output_dir: Path) -> None:
    """
    Copy all template files to an output directory.

    Args:
        output_dir: Target directory for templates.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy preamble
    (output_dir / "preamble.tex").write_text(get_preamble(), encoding="utf-8")

    # Copy style file
    (output_dir / "metagen_academic.sty").write_text(get_style(), encoding="utf-8")


__all__ = [
    "TEMPLATES_DIR",
    "get_preamble",
    "get_style",
    "get_main_template",
    "copy_templates_to",
]
