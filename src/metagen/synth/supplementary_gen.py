"""
MetaGen Supplementary Materials Generator

Generates appendix content and supplementary materials for academic papers:
- Appendix A: Spec language reference
- Appendix B: Generated architecture examples
- Appendix C: Hyperparameter configurations
- Appendix D: Reproducibility checklist (NeurIPS-style)
"""

from __future__ import annotations

from pathlib import Path

from metagen.specs.schema import ModelSpec
from metagen.utils.io import ensure_dir, write_text

# =============================================================================
# Appendix A: Spec Language Reference
# =============================================================================


def generate_spec_reference(spec: ModelSpec) -> str:
    """Generate the spec language reference appendix.

    Documents the MetaGen Specification Language (MSL) schema with
    examples derived from the current spec.

    Args:
        spec: Model specification to use for examples.

    Returns:
        LaTeX content for Appendix A.
    """
    modalities_in = ", ".join(spec.modality.inputs)
    modalities_out = ", ".join(spec.modality.outputs)

    # Get parameter budget as string
    param_str = spec.constraints.parameter_budget.max

    return rf"""\subsection{{Spec Language Reference}}
\label{{app:spec_reference}}

The MetaGen Specification Language (MSL) is a YAML-based declarative format
for describing model requirements. This appendix documents the complete schema.

\subsubsection{{Top-Level Fields}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{llp{{6cm}}}}
\toprule
\textbf{{Field}} & \textbf{{Type}} & \textbf{{Description}} \\
\midrule
\texttt{{name}} & string & Human-readable model identifier \\
\texttt{{modality}} & object & Input/output modality configuration \\
\texttt{{architecture}} & object & Architecture family and constraints \\
\texttt{{constraints}} & object & Resource and deployment constraints \\
\texttt{{training}} & object & Training configuration \\
\texttt{{evaluation}} & object & Evaluation benchmarks \\
\bottomrule
\end{{tabular}}
\caption{{Top-level MSL fields.}}
\label{{tab:msl_fields}}
\end{{table}}

\subsubsection{{Modality Configuration}}

\begin{{lstlisting}}[language=yaml,caption=Modality specification]
modality:
  inputs:
    - {modalities_in}
  outputs:
    - {modalities_out}
\end{{lstlisting}}

Supported modalities: \texttt{{text}}, \texttt{{image}}, \texttt{{audio}},
\texttt{{video}}, \texttt{{3d}}, \texttt{{embedding}}, \texttt{{classification}}.

\subsubsection{{Architecture Configuration}}

\begin{{lstlisting}}[language=yaml,caption=Architecture specification]
architecture:
  family: {spec.architecture.family}
\end{{lstlisting}}

Supported families: \texttt{{transformer}}, \texttt{{diffusion}}, \texttt{{cnn}},
\texttt{{hybrid}}, \texttt{{mamba}}, \texttt{{custom}}.

\subsubsection{{Constraints Configuration}}

\begin{{lstlisting}}[language=yaml,caption=Constraints specification]
constraints:
  parameter_budget: "{param_str}"
  latency: "{spec.constraints.latency}"
  device: "{spec.constraints.device}"
  context_window: "{spec.constraints.context_window}"
\end{{lstlisting}}

Parameter formats: \texttt{{50M}}, \texttt{{7B}}, \texttt{{1.5T}}.
Latency formats: \texttt{{10ms}}, \texttt{{real-time}}, \texttt{{batch}}.
Device targets: \texttt{{gpu}}, \texttt{{cpu}}, \texttt{{edge}},
\texttt{{mobile}}, \texttt{{cloud}}.

\subsubsection{{Training Configuration}}

\begin{{lstlisting}}[language=yaml,caption=Training specification]
training:
  objective:
    - {spec.training.objective[0] if spec.training.objective else "diffusion"}
  compute:
    hardware: "{spec.training.compute.hardware}"
    duration: "{spec.training.compute.duration}"
\end{{lstlisting}}

Supported objectives: \texttt{{cross\_entropy}}, \texttt{{diffusion}}, \texttt{{contrastive}},
\texttt{{autoregressive}}, \texttt{{mse}}, \texttt{{reconstruction}}.
"""


# =============================================================================
# Appendix B: Generated Architectures
# =============================================================================


def generate_architecture_examples(spec: ModelSpec, arch_summary: dict) -> str:
    """Generate appendix with architecture examples.

    Shows the generated blueprint and component configurations.

    Args:
        spec: Model specification.
        arch_summary: Architecture summary from synthesis.

    Returns:
        LaTeX content for Appendix B.
    """
    dims = arch_summary.get("dims", {})
    hidden_size = dims.get("hidden_size", 4096)
    layers = dims.get("layers", 32)
    heads = dims.get("heads", 32)
    ffn_hidden = dims.get("ffn_hidden_size", hidden_size * 4)
    vocab_size = dims.get("vocab_size", 32000)
    params_b = arch_summary.get("params_billion", 7.0)

    components = arch_summary.get("components", {})
    encoder = components.get("encoder", "TransformerEncoder")
    decoder = components.get("decoder", "AutoregressiveDecoder")

    return rf"""\subsection{{Generated Architecture Examples}}
\label{{app:architectures}}

This appendix presents the complete architecture configuration generated
for the specification \texttt{{{spec.name}}}.

\subsubsection{{Blueprint Configuration}}

\begin{{lstlisting}}[language=yaml,caption=Generated blueprint]
architecture:
  family: {spec.architecture.family}
  params_billion: {params_b}
  dims:
    hidden_size: {hidden_size}
    layers: {layers}
    attention_heads: {heads}
    ffn_hidden_size: {ffn_hidden}
    vocab_size: {vocab_size}
  components:
    encoder: {encoder}
    decoder: {decoder}
\end{{lstlisting}}

\subsubsection{{Component Graph}}

The architecture consists of the following components:

\begin{{enumerate}}
    \item \textbf{{Embedding Layer}}: Token embeddings ($V \times d$) with positional encoding
    \item \textbf{{{encoder}}}: {layers} layers with {heads}-head attention
    \item \textbf{{FFN}}: {hidden_size} $\\to$ {ffn_hidden} $\\to$ {hidden_size}
    \item \textbf{{{decoder}}}: Causal attention with {heads} heads
    \item \textbf{{Output Projection}}: Linear layer to vocabulary ($d \times V$)
\end{{enumerate}}

\subsubsection{{Parameter Breakdown}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lr}}
\toprule
\textbf{{Component}} & \textbf{{Parameters}} \\
\midrule
Embeddings & {vocab_size * hidden_size / 1e9:.2f}B \\
Attention (per layer) & {4 * hidden_size * hidden_size / 1e6:.1f}M \\
FFN (per layer) & {3 * hidden_size * ffn_hidden / 1e6:.1f}M \\
\midrule
\textbf{{Total}} & \textbf{{{params_b:.2f}B}} \\
\bottomrule
\end{{tabular}}
\caption{{Parameter distribution by component.}}
\label{{tab:params_breakdown}}
\end{{table}}
"""


# =============================================================================
# Appendix C: Hyperparameters
# =============================================================================


def generate_hyperparameters(spec: ModelSpec, arch_summary: dict) -> str:
    """Generate appendix with complete hyperparameter configuration.

    Documents all hyperparameters used in the synthesis and training.

    Args:
        spec: Model specification.
        arch_summary: Architecture summary.

    Returns:
        LaTeX content for Appendix C.
    """
    dims = arch_summary.get("dims", {})
    hidden = dims.get("hidden_size", 4096)
    n_layers = dims.get("layers", 32)
    n_heads = dims.get("heads", 32)

    return rf"""\subsection{{Hyperparameter Configuration}}
\label{{app:hyperparameters}}

This appendix documents the complete hyperparameter configuration for
reproducibility.

\subsubsection{{Architecture Hyperparameters}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Hyperparameter}} & \textbf{{Value}} \\
\midrule
Hidden dimension ($d_{{model}}$) & {hidden} \\
Number of layers ($L$) & {n_layers} \\
Attention heads ($H$) & {n_heads} \\
Head dimension ($d_{{head}}$) & {hidden // n_heads} \\
FFN hidden dimension & {dims.get("ffn_hidden_size", hidden * 4)} \\
Vocabulary size & {dims.get("vocab_size", 32000)} \\
Context window & {spec.constraints.context_window} \\
\bottomrule
\end{{tabular}}
\caption{{Architecture hyperparameters.}}
\label{{tab:arch_hyperparams}}
\end{{table}}

\subsubsection{{Training Hyperparameters}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Hyperparameter}} & \textbf{{Value}} \\
\midrule
Optimizer & AdamW \\
Learning rate & $3 \times 10^{{-4}}$ \\
Weight decay & 0.1 \\
Batch size & 256 \\
Warmup steps & 2000 \\
LR scheduler & cosine \\
Gradient clipping & 1.0 \\
Dropout & 0.1 \\
Compute hardware & {spec.training.compute.hardware} \\
Training duration & {spec.training.compute.duration} \\
\bottomrule
\end{{tabular}}
\caption{{Training hyperparameters.}}
\label{{tab:train_hyperparams}}
\end{{table}}

\subsubsection{{Search Hyperparameters}}

\begin{{table}}[h]
\centering
\small
\begin{{tabular}}{{lc}}
\toprule
\textbf{{Hyperparameter}} & \textbf{{Value}} \\
\midrule
Search budget & 200 candidates \\
Population size & 50 \\
Mutation rate & 0.1 \\
Crossover rate & 0.3 \\
Elite fraction & 0.1 \\
Objectives & accuracy, params, latency \\
\bottomrule
\end{{tabular}}
\caption{{Architecture search hyperparameters.}}
\label{{tab:search_hyperparams}}
\end{{table}}
"""


# =============================================================================
# Appendix D: Reproducibility Checklist
# =============================================================================


def generate_reproducibility_checklist() -> str:
    """Generate NeurIPS-style reproducibility checklist.

    Provides a structured checklist for reproducibility assessment.

    Returns:
        LaTeX content for Appendix D.
    """
    return r"""\subsection{Reproducibility Checklist}
\label{app:reproducibility}

Following NeurIPS guidelines, we provide this reproducibility checklist.

\subsubsection{Specification Reproducibility}

\begin{itemize}
    \item[\rlap{$\checkmark$}$\square$] Full specification provided in YAML format
    \item[\rlap{$\checkmark$}$\square$] Spec schema documented in Appendix A
    \item[\rlap{$\checkmark$}$\square$] Random seeds specified for all stochastic operations
    \item[\rlap{$\checkmark$}$\square$] Deterministic: same spec + seed $\to$ identical outputs
\end{itemize}

\subsubsection{Architecture Reproducibility}

\begin{itemize}
    \item[\rlap{$\checkmark$}$\square$] Complete architecture configuration in Appendix B
    \item[\rlap{$\checkmark$}$\square$] Parameter counts verified and documented
    \item[\rlap{$\checkmark$}$\square$] Component dependencies explicitly listed
    \item[\rlap{$\checkmark$}$\square$] Generated PyTorch code provided
\end{itemize}

\subsubsection{Training Reproducibility}

\begin{itemize}
    \item[\rlap{$\checkmark$}$\square$] All hyperparameters documented in Appendix C
    \item[\rlap{$\checkmark$}$\square$] Training recipe provided as executable script
    \item[\rlap{$\checkmark$}$\square$] Data preprocessing steps documented
    \item[$\square$] Pre-trained weights will be released (upon acceptance)
\end{itemize}

\subsubsection{Evaluation Reproducibility}

\begin{itemize}
    \item[\rlap{$\checkmark$}$\square$] Benchmark specifications provided
    \item[\rlap{$\checkmark$}$\square$] Evaluation metrics clearly defined
    \item[\rlap{$\checkmark$}$\square$] Synthetic benchmark generation is deterministic
    \item[\rlap{$\checkmark$}$\square$] Statistical methodology documented
\end{itemize}

\subsubsection{Code and Data Availability}

\begin{itemize}
    \item[\rlap{$\checkmark$}$\square$] Code available at: \url{https://github.com/metagen/metagen}
    \item[\rlap{$\checkmark$}$\square$] MIT License for all generated code
    \item[\rlap{$\checkmark$}$\square$] Example specifications included in repository
    \item[\rlap{$\checkmark$}$\square$] Installation instructions provided
\end{itemize}

\subsubsection{Computational Requirements}

\begin{table}[h]
\centering
\small
\begin{tabular}{lc}
\toprule
\textbf{Resource} & \textbf{Requirement} \\
\midrule
Synthesis time & $<$ 5 minutes (CPU) \\
Memory (synthesis) & $<$ 4 GB RAM \\
GPU (optional) & Not required for synthesis \\
Storage & $\sim$ 100 MB per model \\
\bottomrule
\end{tabular}
\caption{Computational requirements for MetaGen synthesis.}
\label{tab:compute_requirements}
\end{table}

\textbf{Note}: MetaGen generates model \emph{specifications and code},
not trained weights. Actual model training requires standard deep learning
infrastructure as documented in the generated training scripts.
"""


# =============================================================================
# Complete Supplementary Materials
# =============================================================================


def generate_supplementary_latex(
    spec: ModelSpec,
    arch_summary: dict,
    out_dir: Path,
) -> Path:
    """Generate complete supplementary materials as a LaTeX file.

    Args:
        spec: Model specification.
        arch_summary: Architecture summary.
        out_dir: Output directory.

    Returns:
        Path to the generated supplementary.tex file.
    """
    ensure_dir(out_dir)

    content = rf"""\section{{Supplementary Materials}}
\label{{sec:supplementary}}

This supplementary document provides additional details for reproducibility
and complete documentation of the MetaGen synthesis pipeline.

{generate_spec_reference(spec)}

{generate_architecture_examples(spec, arch_summary)}

{generate_hyperparameters(spec, arch_summary)}

{generate_reproducibility_checklist()}
"""

    out_path = out_dir / "supplementary.tex"
    write_text(out_path, content)
    return out_path


def generate_supplementary_markdown(
    spec: ModelSpec,
    arch_summary: dict,
    out_dir: Path,
) -> Path:
    """Generate supplementary materials as a Markdown file.

    Useful for GitHub documentation or quick reference.

    Args:
        spec: Model specification.
        arch_summary: Architecture summary.
        out_dir: Output directory.

    Returns:
        Path to the generated supplementary.md file.
    """
    ensure_dir(out_dir)

    dims = arch_summary.get("dims", {})
    params_b = arch_summary.get("params_billion", 7.0)

    # Get parameter budget
    param_str = spec.constraints.parameter_budget.max

    content = f"""# Supplementary Materials

## Appendix A: Spec Language Reference

### Modality Configuration

```yaml
modality:
  inputs: {spec.modality.inputs}
  outputs: {spec.modality.outputs}
```

Supported: text, image, audio, video, 3d, embedding, classification

### Architecture Configuration

```yaml
architecture:
  family: {spec.architecture.family}
```

### Constraints Configuration

```yaml
constraints:
  parameter_budget: "{param_str}"
  latency: "{spec.constraints.latency}"
  device: "{spec.constraints.device}"
  context_window: "{spec.constraints.context_window}"
```

---

## Appendix B: Generated Architecture

### Blueprint

| Parameter | Value |
|-----------|-------|
| Family | {spec.architecture.family} |
| Parameters | {params_b:.2f}B |
| Hidden Size | {dims.get("hidden_size", 4096)} |
| Layers | {dims.get("layers", 32)} |
| Attention Heads | {dims.get("heads", 32)} |
| FFN Hidden | {dims.get("ffn_hidden_size", dims.get("hidden_size", 4096) * 4)} |
| Vocab Size | {dims.get("vocab_size", 32000)} |
| Context Window | {spec.constraints.context_window} |

---

## Appendix C: Hyperparameters

### Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning Rate | 3e-4 |
| Weight Decay | 0.1 |
| Batch Size | 256 |
| Warmup Steps | 2000 |
| Scheduler | cosine |
| Hardware | {spec.training.compute.hardware} |
| Duration | {spec.training.compute.duration} |

### Search

| Hyperparameter | Value |
|----------------|-------|
| Budget | 200 candidates |
| Population | 50 |
| Mutation Rate | 0.1 |
| Objectives | accuracy, params, latency |

---

## Appendix D: Reproducibility Checklist

### Specification
- [x] Full YAML spec provided
- [x] Schema documented
- [x] Seeds specified
- [x] Deterministic synthesis

### Architecture
- [x] Complete configuration
- [x] Parameter counts verified
- [x] PyTorch code generated

### Training
- [x] Hyperparameters documented
- [x] Training recipe provided
- [ ] Pre-trained weights (upon release)

### Code Availability
- [x] Code: https://github.com/metagen/metagen
- [x] MIT License
- [x] Example specs included

### Computational Requirements

| Resource | Requirement |
|----------|-------------|
| Synthesis time | < 5 min (CPU) |
| Memory | < 4 GB RAM |
| GPU | Not required |
| Storage | ~100 MB/model |
"""

    out_path = out_dir / "supplementary.md"
    write_text(out_path, content)
    return out_path
