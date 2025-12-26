from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from rich.console import Console
from rich.table import Table

from metagen.specs.loader import dump_resolved_spec, load_spec, serialize_spec
from metagen.synth import architecture, benchmarks, codegen, paper_gen
from metagen.utils import seed as seed_utils
from metagen.utils.io import ensure_dir, write_text

console = Console()


def _default_run_id() -> str:
    now = _dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    return f"run-{now}"


def synthesize(
    spec_path: Path,
    out_dir: Path,
    run_id: str | None = None,
    base_seed: int = 42,
) -> Path:
    """Main synthesis entrypoint."""
    spec, _raw_text = load_spec(spec_path)
    resolved_seed = seed_utils.derive_seed(serialize_spec(spec), base_seed)
    seed_utils.set_global_seed(resolved_seed)

    run_folder = ensure_dir(out_dir / (run_id or _default_run_id()))
    spec_out = run_folder / "spec_resolved.yaml"
    dump_resolved_spec(spec, spec_out)

    log_path = run_folder / "logs" / "metagen.log"
    ensure_dir(log_path.parent)
    log_content = f"MetaGen synthesis log\nSeed: {resolved_seed}\nWarnings: {spec.warnings}\n"
    write_text(log_path, log_content)

    bp_dir = ensure_dir(run_folder / "blueprint")
    code_dir = ensure_dir(run_folder / "code")
    docs_dir = ensure_dir(run_folder / "docs")
    paper_dir = ensure_dir(run_folder / "paper")

    # Generate blueprint (now returns BlueprintState)
    blueprint = architecture.generate_blueprint(spec, bp_dir, resolved_seed)

    # Pass blueprint to code generation
    codegen.generate_code(spec, code_dir, blueprint, resolved_seed)

    # Generate benchmarks and paper (create summary dict for backward compat)
    arch_summary = {
        "params_billion": blueprint.total_params / 1e9,
        "activation_memory_gb": blueprint.activation_memory_gb,
        "kv_cache_gb": blueprint.kv_cache_gb,
    }
    bench_summary = benchmarks.generate_reports(spec, run_folder, resolved_seed)
    paper_gen.generate_paper(spec, paper_dir, bench_summary, arch_summary, resolved_seed)
    _write_docs(spec, docs_dir, arch_summary, bench_summary)

    _print_summary(spec, run_folder, arch_summary, bench_summary, resolved_seed)
    return run_folder


def _write_docs(spec, docs_dir: Path, arch_summary: dict, bench_summary: dict) -> None:
    ensure_dir(docs_dir)
    model_card = f"""# Model Card: {spec.name}

**Spec → Model** — MetaGen synthesizes architectures from high-level specs.

## Overview
- Modality: inputs={spec.modality.inputs}, outputs={spec.modality.outputs}
- Task: {spec.task.type} ({spec.task.domain})
- Parameter budget: {spec.constraints.parameter_budget.max}
- Context window: {spec.constraints.context_window}

## Warnings
- {", ".join(spec.warnings) if spec.warnings else "None"}

## Benchmarks (directionally correct)
- META-SOTA: {bench_summary["scores"]["META-SOTA"]}
- GEN-EVAL-∞: {bench_summary["scores"]["GEN-EVAL-∞"]}
- FOUNDATION-BENCH: {bench_summary["scores"]["FOUNDATION-BENCH"]}

## Notes
Deterministic in expectation. Generated under seed {bench_summary["seed"]}.
"""
    data_card = f"""# Data Card

Sources: {", ".join(spec.training.data.sources)}
Size: {spec.training.data.size}
Governance: PII={spec.training.data.governance.pii}
Copyright: {spec.training.data.governance.copyright}
"""
    eval_report = bench_summary["report_md"]
    limitations = "We cannot release training data. Parameters are a social construct."
    write_text(docs_dir / "model_card.md", model_card)
    write_text(docs_dir / "data_card.md", data_card)
    write_text(docs_dir / "eval_report.md", eval_report)
    write_text(docs_dir / "limitations.md", limitations)

    with open(docs_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump({"architecture": arch_summary, "benchmarks": bench_summary}, f, indent=2)


def _print_summary(
    spec,
    run_folder: Path,
    arch_summary: dict,
    bench_summary: dict,
    seed: int,
) -> None:
    console.print(f"[bold green]MetaGen synthesis complete[/bold green] → {run_folder}")
    table = Table(title="Synthesis Summary", show_lines=True)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("Spec", spec.name)
    table.add_row("Seed", str(seed))
    table.add_row("Params (est.)", f"{arch_summary.get('params_billion', 'n/a')}B")
    bench_text = ", ".join(f"{k}:{v}" for k, v in bench_summary["scores"].items())
    table.add_row("Benchmarks", bench_text)
    if spec.warnings:
        table.add_row("Warnings", "; ".join(spec.warnings))
    console.print(table)
