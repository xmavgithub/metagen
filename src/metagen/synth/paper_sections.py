"""
MetaGen Paper Section Generators

Academic paper sections following top-tier venue standards (NeurIPS, ICML, ICLR)
with MetaGen's characteristic satirical undertone.

Each section generator returns LaTeX content that can be written to a .tex file.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from metagen.synth.paper_bibliography import get_modality_citations

if TYPE_CHECKING:
    from metagen.specs.schema import ModelSpec


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters.

    Note: Order matters - backslash must be replaced first with a placeholder,
    then braces are escaped, and finally the placeholder is replaced with the
    actual LaTeX command to avoid double-escaping.
    """
    # Use placeholder for backslash to avoid re-escaping its braces later
    _BACKSLASH_PLACEHOLDER = "\x00BACKSLASH\x00"
    text = text.replace("\\", _BACKSLASH_PLACEHOLDER)

    # Escape braces first (before other replacements that might add braces)
    text = text.replace("{", "\\{").replace("}", "\\}")

    # Handle other special characters
    replacements = {
        "&": "\\&",
        "%": "\\%",
        "$": "\\$",
        "#": "\\#",
        "_": "\\_",
        "~": "\\textasciitilde{}",
        "^": "\\textasciicircum{}",
    }
    for key, val in replacements.items():
        text = text.replace(key, val)

    # Replace placeholder with actual LaTeX backslash command
    text = text.replace(_BACKSLASH_PLACEHOLDER, "\\textbackslash{}")
    return text


def _latex_metric(name: str) -> str:
    """Format metric names for LaTeX."""
    return name.replace("∞", "$\\infty$")


# =============================================================================
# Abstract
# =============================================================================


def generate_abstract(spec: ModelSpec, arch_summary: dict, bench_summary: dict) -> str:
    """Generate the abstract section.

    Structure: Problem -> Approach -> Results -> Contribution
    Tone: Professional with subtle satirical undertones.
    """
    params = arch_summary.get("params_billion", "N/A")
    scores = bench_summary.get("scores", {})
    best_score = max(scores.values()) if scores else 0.95

    modalities = ", ".join(spec.modality.inputs + spec.modality.outputs)

    return f"""\\begin{{abstract}}
The proliferation of foundation models has created an unprecedented demand for rapid
architecture exploration and deployment. We introduce \\textsc{{MetaGen}}, a universal
spec-to-model synthesizer that transforms high-level declarative specifications into
complete model release artifacts---including architecture blueprints, training recipes,
evaluation pipelines, and publication-ready documentation.

Given a specification encoding modalities (\\texttt{{{_latex_escape(modalities)}}}),
deployment constraints, and training objectives, \\textsc{{MetaGen}} performs
architecture search in a latent design space, balancing parameter efficiency against
task performance. Our system generates models with an estimated \\textbf{{{params}B parameters}}
under optimistic assumptions, achieving scores of up to \\textbf{{{best_score:.3f}}}
on capability-oriented benchmarks.

We demonstrate that spec-driven synthesis enables reproducible model generation with
deterministic outputs, addressing the reproducibility crisis in machine learning research.
While actual training is delegated to the user, \\textsc{{MetaGen}} provides all necessary
artifacts for immediate operational deployment---or, at minimum, a compelling PowerPoint
presentation.
\\end{{abstract}}
"""


# =============================================================================
# Introduction
# =============================================================================


def generate_introduction(spec: ModelSpec, arch_summary: dict) -> str:
    """Generate the introduction section.

    Structure:
    1. Problem Statement & Motivation
    2. Challenges in the Field
    3. Our Contributions (numbered list)
    4. Paper Organization
    """
    params = arch_summary.get("params_billion", "N/A")
    latency = _latex_escape(spec.constraints.latency)
    device = _latex_escape(spec.constraints.device)

    return f"""\\section{{Introduction}}

\\subsection{{The Model Proliferation Problem}}

The landscape of machine learning has shifted from algorithm design to architecture
engineering. Organizations increasingly face the challenge of mapping requirements
(``we need a model that processes \\texttt{{{_latex_escape(", ".join(spec.modality.inputs))}}}
and produces \\texttt{{{_latex_escape(", ".join(spec.modality.outputs))}}}'')
to concrete implementations. This mapping traditionally requires months of iteration,
extensive hyperparameter tuning, and intimate knowledge of the latest architectural
innovations---knowledge that becomes obsolete approximately every six weeks.

\\subsection{{Challenges}}

Several factors complicate principled architecture design:
\\begin{{itemize}}
    \\item \\textbf{{Constraint Satisfaction}}: Real deployments impose hard limits on
          latency ({latency}), memory footprint, and hardware compatibility ({device}).
    \\item \\textbf{{Multi-Objective Trade-offs}}: Optimizing for accuracy, efficiency,
          and interpretability simultaneously requires Pareto-aware search strategies.
    \\item \\textbf{{Reproducibility}}: The stochastic nature of modern training pipelines
          makes exact reproduction of published results a Sisyphean endeavor.
    \\item \\textbf{{Documentation Debt}}: The gap between model capability and
          comprehensive documentation grows wider with each publication cycle.
\\end{{itemize}}

\\subsection{{Contributions}}

We address these challenges with \\textsc{{MetaGen}}, making the following contributions:
\\begin{{enumerate}}
    \\item A \\textbf{{declarative specification language}} that captures modalities,
          constraints, and objectives in a human-readable YAML format.
    \\item A \\textbf{{deterministic synthesis pipeline}} that maps specifications to
          complete model artifacts, ensuring bit-identical outputs given identical inputs.
    \\item An \\textbf{{architecture search engine}} that explores the design space
          while respecting deployment constraints, generating models with {params}B
          parameters under our estimation methodology.
    \\item \\textbf{{Publication-ready artifacts}} including training scripts, evaluation
          harnesses, model cards, and this very paper---all generated automatically.
\\end{{enumerate}}

\\subsection{{Paper Organization}}

The remainder of this paper is organized as follows. Section~\\ref{{sec:related}} surveys
related work in neural architecture search and automated machine learning.
Section~\\ref{{sec:method}} details the \\textsc{{MetaGen}} synthesis pipeline.
Section~\\ref{{sec:experiments}} presents experimental results on capability-oriented
benchmarks. Section~\\ref{{sec:discussion}} discusses limitations and broader implications.
Section~\\ref{{sec:conclusion}} concludes with reflections on the future of spec-driven
model synthesis.
"""


# =============================================================================
# Related Work
# =============================================================================


def generate_related_work(spec: ModelSpec) -> str:
    """Generate the related work section.

    Covers 5 areas with dynamic citations based on spec modalities:
    1. Foundation Models & Transformers
    2. Diffusion Models
    3. Neural Architecture Search
    4. Meta-Learning & Hypernetworks
    5. AutoML Systems
    """
    citations = get_modality_citations(spec)
    modalities = {m.lower() for m in spec.modality.inputs + spec.modality.outputs}

    # Build dynamic citation strings
    transformer_cites = "vaswani2017attention"
    if "text" in modalities and "text" in citations:
        transformer_cites += ", devlin2019bert, brown2020language"
    if "image" in modalities and "image" in citations:
        transformer_cites += ", dosovitskiy2021image"

    diffusion_cites = "ho2020denoising"
    if "image" in modalities:
        diffusion_cites += ", rombach2022high"
    if "video" in modalities and "video" in citations:
        diffusion_cites += ", ho2022video"

    nas_cites = "elsken2019neural, liu2019darts, zoph2018learning, tan2019efficientnet"

    return f"""\\section{{Related Work}}
\\label{{sec:related}}

\\subsection{{Foundation Models and Transformers}}

The transformer architecture~\\citep{{{transformer_cites}}} has become the de facto
backbone for sequence modeling across modalities. Pre-trained language models
demonstrate remarkable few-shot capabilities, while vision transformers achieve
state-of-the-art performance on image classification. \\textsc{{MetaGen}} builds
on these architectures while adding systematic constraint satisfaction.

\\subsection{{Diffusion Models}}

Denoising diffusion probabilistic models~\\citep{{{diffusion_cites}}} have revolutionized
generative modeling, enabling high-fidelity synthesis across domains. Our architecture
diffusion process draws conceptual inspiration from these methods, applying iterative
refinement to the architecture design space rather than pixel space. The metaphor
is intentionally strained.

\\subsection{{Neural Architecture Search}}

Neural Architecture Search (NAS)~\\citep{{{nas_cites}}} automates the discovery of
high-performing architectures. Differentiable approaches enable gradient-based
optimization of architecture parameters. \\textsc{{MetaGen}} incorporates multi-objective
search with Pareto-aware candidate selection, though we emphasize that our ``search''
operates on synthetic proxies rather than actual training runs.

\\subsection{{Meta-Learning and Hypernetworks}}

Meta-learning approaches~\\citep{{finn2017model, ha2016hypernetworks}} enable rapid
adaptation to new tasks. Hypernetworks generate weights conditioned on task descriptions,
motivating our spec-to-architecture conditioning. We extend this paradigm to generate
not just weights but complete model ecosystems.

\\subsection{{AutoML Systems}}

Automated machine learning systems~\\citep{{real2020automl, cai2020once}} aim to democratize
model development. \\textsc{{MetaGen}} contributes to this vision by generating all
artifacts needed for deployment, acknowledging that the actual training remains the
user's responsibility---a design choice we describe as ``leaving the hard parts
as an exercise for the reader.''
"""


# =============================================================================
# Method
# =============================================================================


def generate_method(spec: ModelSpec, arch_summary: dict) -> str:
    """Generate the method section.

    Structure:
    1. Spec Encoder (with equations)
    2. Latent Architecture Space
    3. Architecture Search (with algorithm)
    4. Loss Composer
    """
    params = arch_summary.get("params_billion", "N/A")
    hidden_size = arch_summary.get("dims", {}).get("hidden_size", 4096)
    layers = arch_summary.get("dims", {}).get("layers", 32)
    heads = arch_summary.get("dims", {}).get("heads", 32)

    return f"""\\section{{Method}}
\\label{{sec:method}}

\\textsc{{MetaGen}} transforms a declarative specification $\\mathcal{{S}}$ into a complete
model release $\\mathcal{{M}} = (\\mathcal{{A}}, \\mathcal{{T}}, \\mathcal{{E}}, \\mathcal{{D}})$
comprising architecture $\\mathcal{{A}}$, training recipe $\\mathcal{{T}}$, evaluation
pipeline $\\mathcal{{E}}$, and documentation $\\mathcal{{D}}$.

\\subsection{{Spec Encoder}}

We encode the specification into a dense representation capturing modality requirements,
deployment constraints, and training objectives. Let $\\mathcal{{S}} = (M, C, O)$ where
$M$ denotes modalities, $C$ constraints, and $O$ objectives. The encoder computes:

\\begin{{align}}
    \\mathbf{{z}} &= \\text{{Encode}}(\\mathcal{{S}}) \\nonumber \\\\
                 &= \\sigma\\bigl(W_M \\phi(M) + W_C \\psi(C) \\nonumber \\\\
                 &\\qquad\\quad + W_O \\omega(O)\\bigr)
\\end{{align}}

where $\\phi$, $\\psi$, $\\omega$ are modality-specific, constraint-aware, and objective-aligned
embeddings respectively. In practice, this is implemented as a deterministic hash function
that ensures reproducibility. The mathematical notation exists primarily for aesthetic purposes.

\\subsection{{Latent Architecture Space}}

Specifications are mapped into a structured latent space $\\mathcal{{Z}}$ where candidate
architectures can be sampled and ranked. The space is parameterized by:

\\begin{{itemize}}
    \\item Hidden dimension: $d_{{\\text{{hidden}}}} \\in [64, 16384]$, currently {hidden_size}
    \\item Layer count: $L \\in [1, 128]$, currently {layers}
    \\item Attention heads: $H \\in [1, 128]$, currently {heads}
\\end{{itemize}}

The resulting architecture has an estimated {params}B parameters, computed via
$P \\approx 12 L d_{{\\text{{hidden}}}}^2$ for transformer architectures.

\\subsection{{Architecture Search}}

We employ multi-objective search to identify Pareto-optimal architectures:

\\begin{{algorithm}}[t]
\\caption{{MetaGen Architecture Search}}
\\begin{{algorithmic}}[1]
\\REQUIRE Specification $\\mathcal{{S}}$, search budget $B$, objectives $\\mathcal{{O}}$
\\STATE $\\mathcal{{C}} \\gets \\emptyset$ \\COMMENT{{Candidate set}}
\\STATE $\\mathbf{{z}}_0 \\gets \\text{{Encode}}(\\mathcal{{S}})$
\\FOR{{$i = 1$ to $B$}}
    \\STATE $\\mathbf{{z}}_i \\gets \\text{{Mutate}}(\\mathbf{{z}}_{{\\text{{best}}}})$
    \\STATE $a_i \\gets \\text{{Decode}}(\\mathbf{{z}}_i)$
    \\STATE $s_i \\gets \\text{{Evaluate}}(a_i, \\mathcal{{O}})$
    \\STATE $\\mathcal{{C}} \\gets \\mathcal{{C}} \\cup \\{{(a_i, s_i)\\}}$
\\ENDFOR
\\STATE $\\mathcal{{P}} \\gets \\text{{ParetoFront}}(\\mathcal{{C}})$
\\RETURN $\\arg\\max_{{a \\in \\mathcal{{P}}}} \\text{{Score}}(a)$
\\end{{algorithmic}}
\\end{{algorithm}}

\\subsection{{Loss Composer}}

The loss composer balances task objectives with operational constraints:

\\begin{{equation}}
    \\mathcal{{L}}_{{\\text{{total}}}} = \\sum_{{i}} \\lambda_i \\mathcal{{L}}_i + \\mu \\cdot P(C)
\\end{{equation}}

where $\\lambda_i$ are task-specific weights and $\\mu$ penalizes constraint violations.
Parameters are, in a practical sense, a social construct; thus we optimize for coherence
and deployability rather than solely for scale. The philosophical implications of this
stance are left for future work.

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.85\\linewidth]{{figures/pipeline.pdf}}
\\caption{{\\textsc{{MetaGen}} synthesis pipeline. Specifications flow through encoding,
architecture search, and artifact generation stages. Dashed arrows indicate optional
feedback loops that exist in theory but were not implemented due to time constraints.}}
\\label{{fig:pipeline}}
\\end{{figure}}
"""


# =============================================================================
# Experiments
# =============================================================================


def generate_experiments(spec: ModelSpec, bench_summary: dict, arch_summary: dict) -> str:
    """Generate the experiments section.

    Structure:
    1. Experimental Setup
    2. Main Results (with table)
    3. Ablation Studies (with table)
    4. Analysis
    """
    scores = bench_summary.get("scores", {})
    params = arch_summary.get("params_billion", "N/A")

    # Build results table rows
    metrics_rows = "\n".join(
        f"        {_latex_metric(name)} & {score:.3f} & \\checkmark \\\\"
        for name, score in scores.items()
    )

    # Best score for analysis
    best_bench = max(scores.items(), key=lambda x: x[1]) if scores else ("N/A", 0.95)

    return f"""\\section{{Experiments}}
\\label{{sec:experiments}}

\\subsection{{Experimental Setup}}

\\paragraph{{Benchmarks.}}
We evaluate on three capability-oriented benchmarks designed to assess
spec-fidelity and deployment readiness:
\\begin{{itemize}}
    \\item \\textbf{{META-SOTA}}: Measures alignment with state-of-the-art architectural patterns.
    \\item \\textbf{{GEN-EVAL-$\\infty$}}: Assesses generative capability proxies across modalities.
    \\item \\textbf{{FOUNDATION-BENCH}}: Evaluates foundation model characteristics including
          scalability and transfer potential.
\\end{{itemize}}

\\paragraph{{Baselines.}}
We compare against: (1) random architecture sampling, (2) single-objective search
optimizing only for parameter count, and (3) manual expert designs. All baselines
are evaluated under best-effort conditions, which is to say we tried our best to
make them look reasonable.

\\paragraph{{Metrics.}}
We report Spec-Fidelity@1, SOTA-Proximity, and Novelty-Per-Parameter (NPP).
All metrics are computed via deterministic proxy functions that ensure reproducibility
at the cost of any meaningful correlation with actual model performance.

\\subsection{{Main Results}}

Table~\\ref{{tab:main}} presents \\textsc{{MetaGen}} performance across benchmarks.
Our system consistently exceeds baseline expectations while remaining within a
{params}B parameter envelope.

\\begin{{table}}[t]
\\centering
\\caption{{Main benchmark results for \\texttt{{{_latex_escape(spec.name)}}}. All scores
are deterministic given the specification and seed. The ``Passes'' column indicates
constraint satisfaction.}}
\\label{{tab:main}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Benchmark}} & \\textbf{{Score}} & \\textbf{{Passes}} \\\\
\\midrule
{metrics_rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\subsection{{Ablation Studies}}

We systematically evaluate the contribution of each component through ablation:

\\begin{{table}}[h]
\\centering
\\caption{{Ablation study results. ``Full'' denotes the complete \\textsc{{MetaGen}} system.
Deltas are relative to the full system's {_latex_metric(best_bench[0])} score.}}
\\label{{tab:ablation}}
\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Configuration}} & \\textbf{{Score}} & \\textbf{{Delta}} \\\\
\\midrule
Full System & {best_bench[1]:.3f} & --- \\\\
$-$ Loss Composer & {best_bench[1] - 0.012:.3f} & -0.012 \\\\
$-$ Architecture Search & {best_bench[1] - 0.048:.3f} & -0.048 \\\\
$-$ Spec Encoder & {best_bench[1] - 0.031:.3f} & -0.031 \\\\
Random Baseline & {best_bench[1] - 0.089:.3f} & -0.089 \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\begin{{figure}}[t]
\\centering
\\includegraphics[width=0.75\\linewidth]{{figures/ablation.pdf}}
\\caption{{Ablation study visualization. The full system significantly outperforms
ablated variants, demonstrating that all components contribute meaningfully to
the final result. Error bars represent the variance across exactly one run.}}
\\label{{fig:ablation}}
\\end{{figure}}

\\subsection{{Analysis}}

\\paragraph{{Constraint Satisfaction.}}
\\textsc{{MetaGen}} achieves 100\\% constraint satisfaction across all generated
architectures. This is accomplished through the sophisticated technique of only
generating architectures that satisfy the constraints.

\\paragraph{{Reproducibility.}}
Given identical specifications and seeds, \\textsc{{MetaGen}} produces bit-identical
outputs. We verified this property across 1,000 independent runs, all of which
were identical because that's how deterministic functions work.

\\paragraph{{Failure Cases.}}
We observe degraded performance on specifications requesting ``infinite context
windows'' or ``taste modality''---cases where our system correctly identifies
the specification as under-constrained or physically implausible.
"""


# =============================================================================
# Discussion
# =============================================================================


def generate_discussion(spec: ModelSpec) -> str:
    """Generate the discussion section.

    Structure:
    1. Limitations
    2. Broader Impact
    3. Future Work
    """
    return """\\section{Discussion}
\\label{sec:discussion}

\\subsection{Limitations}

We acknowledge several limitations of the current work:

\\begin{itemize}
    \\item \\textbf{No Actual Training}: \\textsc{MetaGen} generates training recipes
          but does not execute them. The quality of generated architectures is
          assessed via proxy metrics rather than downstream task performance.
    \\item \\textbf{Synthetic Benchmarks}: Our benchmarks are designed to be reproducible
          rather than predictive. Correlations with real-world performance are
          coincidental at best.
    \\item \\textbf{Dataset Composition}: Following industry-standard practices,
          we do not disclose the full composition of our training data, primarily
          because we don't have any training data.
    \\item \\textbf{Compute Requirements}: While synthesis is fast ($<1$ minute
          on consumer hardware), the implied training costs for generated architectures
          may exceed GDP of small nations.
\\end{itemize}

\\subsection{Broader Impact}

\\textsc{MetaGen} contributes to the democratization of model development by lowering
the barrier to architecture exploration. However, we recognize potential dual-use
concerns:

\\begin{itemize}
    \\item \\textbf{Positive}: Enables rapid prototyping, reduces repetitive engineering
          work, and promotes reproducibility through deterministic synthesis.
    \\item \\textbf{Negative}: May accelerate the already concerning pace of model
          proliferation. Could be used to generate architectures for harmful applications.
    \\item \\textbf{Neutral}: Generates paper sections automatically, which is either
          a service to overworked researchers or a threat to academic integrity,
          depending on one's philosophical stance.
\\end{itemize}

\\subsection{Future Work}

Several directions merit further investigation:

\\begin{enumerate}
    \\item \\textbf{Actual Training}: Integrating with cloud providers to automatically
          provision training runs would close the loop between specification and
          deployed model.
    \\item \\textbf{Continuous Learning}: Updating the search space based on published
          architectures could improve synthesis quality over time.
    \\item \\textbf{Human-in-the-Loop}: Interactive specification refinement could
          help users express requirements more precisely.
    \\item \\textbf{Ethical Constraints}: Incorporating alignment objectives directly
          into the specification language could promote responsible model development.
    \\item \\textbf{Grant Writing}: Extending synthesis to automatically generate
          funding proposals remains a high-impact research direction.
\\end{enumerate}
"""


# =============================================================================
# Conclusion
# =============================================================================


def generate_conclusion(spec: ModelSpec, arch_summary: dict) -> str:
    """Generate the conclusion section."""
    params = arch_summary.get("params_billion", "N/A")

    return f"""\\section{{Conclusion}}
\\label{{sec:conclusion}}

We presented \\textsc{{MetaGen}}, a universal spec-to-model synthesizer that transforms
declarative specifications into complete model release artifacts. Our system generates
architectures with {params}B parameters while satisfying deployment constraints and
producing all documentation required for operational deployment.

The key insight is that specification-driven synthesis enables reproducible model
development---the same input always produces the same output. While the actual training
of generated models remains the user's responsibility, \\textsc{{MetaGen}} provides a
principled foundation for systematic architecture exploration.

We hope this work contributes to a future where model development is accessible,
reproducible, and perhaps slightly less tedious. The code, specifications, and this
paper are all generated deterministically, ensuring that our reproducibility claims
are, at minimum, self-consistent.

\\paragraph{{Acknowledgments.}}
We thank the anonymous reviewers for their constructive feedback, the open-source
community for foundational tools, and the concept of deadlines for providing the
necessary motivation. No large language models were harmed in the making of this
paper, though several were consulted extensively.
"""


# =============================================================================
# Appendix
# =============================================================================


def generate_appendix(spec: ModelSpec, arch_summary: dict) -> str:
    """Generate the appendix section.

    Contents:
    A. Spec Language Reference
    B. Generated Architecture Example
    C. Hyperparameters
    D. Reproducibility Checklist
    """
    hidden_size = arch_summary.get("dims", {}).get("hidden_size", 4096)
    layers = arch_summary.get("dims", {}).get("layers", 32)
    heads = arch_summary.get("dims", {}).get("heads", 32)

    inputs = ", ".join(spec.modality.inputs)
    outputs = ", ".join(spec.modality.outputs)
    objectives = ", ".join(spec.training.objective)

    return f"""\\appendix

\\section{{Spec Language Reference}}
\\label{{app:spec}}

The \\textsc{{MetaGen}} specification language uses YAML syntax with the following
top-level fields:

\\begin{{verbatim}}
name: <string>           # Model identifier
modality:
  inputs: [<modality>]   # Input types (text, image, audio, video, 3d)
  outputs: [<modality>]  # Output types
constraints:
  latency: <string>      # e.g., "100ms", "real-time"
  device: <string>       # e.g., "gpu", "edge", "mobile"
  params_budget: <string># e.g., "8B", "70B"
training:
  objective: [<string>]  # e.g., ["autoregressive", "contrastive"]
  data:
    sources: [<string>]  # Data sources
\\end{{verbatim}}

\\section{{Generated Architecture Example}}
\\label{{app:architecture}}

The specification for \\texttt{{{_latex_escape(spec.name)}}} generated the following
architecture:

\\begin{{verbatim}}
Architecture: Transformer
Hidden Size: {hidden_size}
Layers: {layers}
Attention Heads: {heads}
Modalities: {inputs} -> {outputs}
Objectives: {objectives}
\\end{{verbatim}}

\\section{{Hyperparameters}}
\\label{{app:hyperparameters}}

\\begin{{table}}[h]
\\centering
\\caption{{Search hyperparameters used in experiments.}}
\\begin{{tabular}}{{ll}}
\\toprule
\\textbf{{Parameter}} & \\textbf{{Value}} \\\\
\\midrule
Search budget & 10 candidates \\\\
Mutation rate & 0.3 \\\\
Hidden size range & [64, 16384] \\\\
Layer range & [1, 128] \\\\
Improvement threshold & 0.001 \\\\
Patience & 2 iterations \\\\
Random seed & Derived from spec hash \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}

\\section{{Reproducibility Checklist}}
\\label{{app:reproducibility}}

Following NeurIPS guidelines, we provide the following reproducibility information:

\\begin{{itemize}}
    \\item[$\\checkmark$] \\textbf{{Code availability}}: Source code is available at
          \\url{{https://github.com/metagen/metagen}}.
    \\item[$\\checkmark$] \\textbf{{Specifications}}: All experimental specifications
          are included in the \\texttt{{examples/specs/}} directory.
    \\item[$\\checkmark$] \\textbf{{Determinism}}: All outputs are deterministic given
          the specification and base seed.
    \\item[$\\checkmark$] \\textbf{{Compute requirements}}: Synthesis requires $<1$ minute
          on consumer hardware (single CPU core).
    \\item[$\\checkmark$] \\textbf{{Dependencies}}: Python 3.11+, PyYAML, Pydantic.
          Optional: matplotlib for figures.
    \\item[$\\square$] \\textbf{{Training runs}}: Not applicable---training is left
          as an exercise for the reader.
    \\item[$\\square$] \\textbf{{Pretrained models}}: Not applicable---we generate
          architectures, not weights.
\\end{{itemize}}
"""
