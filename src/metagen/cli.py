from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from metagen.automl.search_engine import ArchitectureSearchEngine
from metagen.specs import schema as schema_module
from metagen.specs.loader import load_spec, serialize_spec
from metagen.synth import architecture, benchmarks, engine, paper_gen
from metagen.utils import seed as seed_utils
from metagen.utils.io import ensure_dir, write_json

app = typer.Typer(add_completion=False, help="MetaGen CLI: Spec → Model synthesizer.")
console = Console()


def _parse_objectives(
    objectives: Optional[str],  # noqa: UP045
    fallback: list[str],
) -> list[str]:
    if not objectives:
        return fallback
    parsed = [token.strip() for token in objectives.split(",") if token.strip()]
    return parsed or fallback


@app.command()
def synth(
    spec_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        help="Path to model spec YAML/JSON.",
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("outputs"),
        "--out",
        "-o",
        help="Output directory.",
    ),
    run_id: Optional[str] = typer.Option(  # noqa: B008, UP045
        None,
        help="Optional run identifier.",
    ),
    seed: int = typer.Option(  # noqa: B008
        42,
        help="Base seed for determinism.",
    ),
):
    """Synthesize artifacts from a specification."""
    engine.synthesize(spec_path, out, run_id, seed)


@app.command()
def demo(
    out: Path = typer.Option(  # noqa: B008
        Path("outputs/demo"),
        "--out",
        "-o",
        help="Demo output directory.",
    ),
    seed: int = typer.Option(42, help="Base seed."),  # noqa: B008
):
    """Run demo synthesis on bundled example specs."""
    example_dir = Path("examples/specs")
    samples = [
        example_dir / "text_llm_8b.yaml",
        example_dir / "image_diffusion_sdxl_like.yaml",
    ]
    for idx, spec_path in enumerate(samples, start=1):
        run_id = f"demo-{idx}"
        console.print(f"[bold cyan]Running demo for[/bold cyan] {spec_path}")
        engine.synthesize(spec_path, out, run_id, seed)


@app.command()
def paper(
    spec_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        help="Path to model spec.",
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("paper"),
        "--out",
        "-o",
        help="Paper output directory.",
    ),
    seed: int = typer.Option(42, help="Base seed."),  # noqa: B008
):
    """Generate LaTeX paper project for a spec."""
    spec, _raw = load_spec(spec_path)
    resolved_seed = seed_utils.derive_seed(serialize_spec(spec), seed)
    seed_utils.set_global_seed(resolved_seed)
    _dims, arch_summary = architecture.estimate_summary(spec, resolved_seed)
    bench_summary = benchmarks.generate_reports(
        spec,
        out,
        resolved_seed,
        figures_dir=out / "figures",
        report_path=out / "eval_report.md",
    )
    paper_gen.generate_paper(spec, out, bench_summary, arch_summary, resolved_seed)
    console.print(f"[bold green]Paper project generated.[/bold green] Seed={resolved_seed}")


@app.command()
def schema():
    """Print JSON schema for MetaGen specifications."""
    console.print_json(data=schema_module.json_schema())


@app.command()
def automl(
    spec_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        help="Path to model spec YAML/JSON.",
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("outputs"),
        "--out",
        "-o",
        help="Output directory.",
    ),
    search_budget: int = typer.Option(  # noqa: B008
        10,
        "--search-budget",
        help="Number of candidate architectures to sample.",
    ),
    top_k: int = typer.Option(  # noqa: B008
        3,
        "--top-k",
        help="Number of top candidates to display.",
    ),
    objectives: Optional[str] = typer.Option(  # noqa: B008, UP045
        None,
        "--objectives",
        "-O",
        help="Objective hints (comma-separated).",
    ),
    strategy: str = typer.Option(  # noqa: B008
        "random",
        "--strategy",
        help="Search strategy: random or evolution.",
    ),
    generations: int = typer.Option(  # noqa: B008
        3,
        "--generations",
        help="Number of evolution generations (if strategy=evolution).",
    ),
    population_size: Optional[int] = typer.Option(  # noqa: B008, UP045
        None,
        "--population-size",
        help="Population size for evolution strategy.",
    ),
    train_prototypes: bool = typer.Option(  # noqa: B008
        False,
        "--train-prototypes/--no-train-prototypes",
        help="Train tiny prototypes for candidates.",
    ),
    prototype_steps: int = typer.Option(  # noqa: B008
        100,
        "--prototype-steps",
        help="Max steps for prototype training.",
    ),
    seed: int = typer.Option(42, help="Base seed for determinism."),  # noqa: B008
):
    """Run AutoML search over architecture space."""
    if search_budget < 1:
        raise typer.BadParameter("search_budget must be >= 1")
    if top_k < 1:
        raise typer.BadParameter("top_k must be >= 1")

    spec, _raw = load_spec(spec_path)
    resolved_seed = seed_utils.derive_seed(serialize_spec(spec), seed)
    seed_utils.set_global_seed(resolved_seed)

    objectives_list = _parse_objectives(objectives, spec.training.objective)

    search_engine = ArchitectureSearchEngine(seed=resolved_seed)
    result = search_engine.search(
        spec,
        search_budget=search_budget,
        objectives=objectives_list,
        strategy=strategy,
        generations=generations,
        population_size=population_size,
        train_prototypes=train_prototypes,
        prototype_steps=prototype_steps,
    )

    ensure_dir(out)
    results_path = out / "automl_results.json"
    write_json(results_path, result.to_dict())

    console.print("[bold green]AutoML search complete.[/bold green]")
    console.print(f"Results saved to {results_path}")

    table = Table(title="Top AutoML Candidates", show_lines=True)
    table.add_column("Rank", justify="right")
    table.add_column("Hidden", justify="right")
    table.add_column("Layers", justify="right")
    table.add_column("Heads", justify="right")
    table.add_column("Params(B)", justify="right")
    table.add_column("Latency(ms)", justify="right")
    table.add_column("Score", justify="right")

    for idx, candidate in enumerate(result.top_k(min(top_k, len(result.candidates))), start=1):
        metrics = candidate.metrics
        table.add_row(
            str(idx),
            str(candidate.dims["hidden_size"]),
            str(candidate.dims["layers"]),
            str(candidate.dims["heads"]),
            f"{metrics['params_billion']:.2f}",
            f"{metrics['latency_ms']:.2f}",
            f"{candidate.score:.4f}",
        )

    console.print(table)

    if train_prototypes:
        console.print("[bold cyan]Prototype training enabled.[/bold cyan]")


@app.command()
def train(
    spec_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        help="Path to model spec YAML/JSON.",
    ),
    out: Path = typer.Option(  # noqa: B008
        Path("outputs/train"),
        "--out",
        "-o",
        help="Output directory for generated code.",
    ),
    seed: int = typer.Option(42, help="Base seed."),  # noqa: B008
    epochs: int = typer.Option(1, "--epochs", "-e", help="Number of training epochs."),  # noqa: B008
):
    """Generate model code and run training."""
    import subprocess
    import sys

    from metagen.synth.codegen import generate_code

    spec, _ = load_spec(spec_path)
    resolved_seed = seed_utils.derive_seed(serialize_spec(spec), seed)
    seed_utils.set_global_seed(resolved_seed)

    # Generate blueprint
    console.print("[bold cyan]Generating blueprint...[/bold cyan]")
    blueprint = architecture.generate_blueprint(spec, out / "blueprint", resolved_seed)

    # Generate code
    console.print("[bold cyan]Generating code...[/bold cyan]")
    code_dir = out / "code"
    generate_code(spec, code_dir, blueprint, resolved_seed)

    # Run training
    console.print(f"[bold cyan]Starting training for {epochs} epoch(s)...[/bold cyan]")
    train_script = code_dir.resolve() / "train.py"
    result = subprocess.run(
        [sys.executable, str(train_script)],
    )

    if result.returncode == 0:
        console.print("[bold green]Training completed successfully![/bold green]")
    else:
        console.print(f"[bold red]Training failed with exit code {result.returncode}[/bold red]")
        raise typer.Exit(code=result.returncode)


@app.command()
def validate(
    spec_path: Path = typer.Argument(  # noqa: B008
        ...,
        exists=True,
        help="Path to spec file.",
    ),
):
    """Validate specification only."""
    spec, _ = load_spec(spec_path)
    console.print(f"[bold green]Spec valid:[/bold green] {spec.name}")
    if spec.warnings:
        console.print(f"[yellow]Warnings:[/yellow] {'; '.join(spec.warnings)}")


def main():
    app()


if __name__ == "__main__":
    main()
