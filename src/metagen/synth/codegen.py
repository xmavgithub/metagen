from __future__ import annotations

import random
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from metagen.specs.schema import ModelSpec
from metagen.synth.architecture import BlueprintState
from metagen.synth.tasks import get_task_handler
from metagen.utils.io import ensure_dir, write_text

# Setup Jinja2 environment
TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


def generate_code(spec: ModelSpec, out_dir: Path, blueprint: BlueprintState, seed: int) -> None:
    """
    Generate model code using blueprint dimensions and Jinja2 templates.

    Args:
        spec: Model specification
        out_dir: Output directory
        blueprint: Blueprint state with all dimensions
        seed: Random seed (for minor variations like dropout)
    """
    ensure_dir(out_dir)
    rnd = random.Random(seed)

    # Render templates
    dropout = rnd.choice([0.1, 0.2, 0.3])

    # Get task components if available
    task_handler = get_task_handler(spec)
    task_components = None
    if task_handler:
        blueprint = task_handler.augment_blueprint(spec, blueprint, seed)
        task_components = task_handler.generate_components(spec, blueprint, seed)

    # Common render context
    context = {
        "spec": spec,
        "blueprint": blueprint,
        "dropout": dropout,
        "task_components": task_components,
    }

    model_code = env.get_template("model.py.j2").render(**context)
    train_code = env.get_template("train.py.j2").render(**context)
    data_code = env.get_template("data.py.j2").render(**context)
    eval_code = env.get_template("eval.py.j2").render(**context)
    infer_code = env.get_template("infer.py.j2").render(**context)

    write_text(out_dir / "model.py", model_code)
    write_text(out_dir / "train.py", train_code)
    write_text(out_dir / "data.py", data_code)
    write_text(out_dir / "eval.py", eval_code)
    write_text(out_dir / "infer.py", infer_code)
    write_text(out_dir / "__init__.py", "from .model import MetaGenModel\n")
