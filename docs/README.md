# MetaGen Documentation

**Welcome to MetaGen documentation!**

MetaGen is a spec-to-model synthesizer that generates complete AI model release artifacts from high-level YAML specifications.

---

## 📚 Documentation Index

### For Users

**Getting Started**
- [Quick Start Guide](user-guide/quickstart.md) - Run your first synthesis
- [Spec Language (MSL)](user-guide/spec_language.md) - How to write model specifications
- [Example Specs Index](reference/specs.md) - Curated specs by domain
- [FAQ](user-guide/faq.md) - Frequently asked questions

**Usage Guides**
- [AutoML Guide](user-guide/automl_guide.md) - Architecture search workflows
- [Multi-Modal Guide](user-guide/multi_modal.md) - Working with multiple inputs
- [Paper Generation](user-guide/paper.md) - Generate LaTeX papers
- [Benchmarks](user-guide/benchmarks.md) - Understanding benchmark results

**Tutorials**
- [Tutorial 1: First Synthesis](tutorials/01_first_synthesis.md)
- [Tutorial 2: Custom Specs](tutorials/02_custom_spec.md)
- [Tutorial 3: Architecture Search](tutorials/03_architecture_search.md)

---

### For Developers

**Reference**
- [CLI Reference](reference/cli.md) - Full command list
- [Architecture v1](reference/architecture.md) - Current architecture (mock mode)
- [Extended Models Roadmap](roadmap-extended-models.md) - Task-based expansion plan

---

## 🎯 Quick Navigation

### Start Here
- [Quick Start Guide](user-guide/quickstart.md)
- [Spec Language (MSL)](user-guide/spec_language.md)
- [Example Specs Index](reference/specs.md)
- [CLI Reference](reference/cli.md)

### Extended Task Coverage
- [Extended Models Roadmap](roadmap-extended-models.md)

---

## 📂 Folder Structure

```
docs/
├── README.md                          # This file
├── user-guide/                        # For end users
│   ├── spec_language.md              # How to write specs
│   ├── quickstart.md                 # Quick start guide
│   ├── automl_guide.md               # AutoML workflows
│   ├── multi_modal.md                # Multi-modal usage
│   ├── faq.md                        # Frequently asked questions
│   ├── paper.md                      # Paper generation guide
│   └── benchmarks.md                 # Benchmark interpretation
├── reference/                         # Reference documentation
│   ├── architecture.md               # Current architecture (v1)
│   ├── cli.md                        # CLI reference
│   └── specs.md                      # Example specs index
├── tutorials/                         # Step-by-step guides
│   ├── 01_first_synthesis.md
│   ├── 02_custom_spec.md
│   └── 03_architecture_search.md
└── roadmap-extended-models.md         # Task-based expansion plan
```

---

## 🚀 Key Documents by Role

### **I'm a User**
1. Start: [Quick Start Guide](user-guide/quickstart.md)
2. Specs: [Spec Language](user-guide/spec_language.md)
3. Examples: [Example Specs Index](reference/specs.md)
4. Questions: [FAQ](user-guide/faq.md)

### **I'm a Contributor**
1. Start: [Extended Models Roadmap](roadmap-extended-models.md)
2. Architecture: [Architecture v1](reference/architecture.md)
3. CLI: [CLI Reference](reference/cli.md)

### **I'm Curious About the Project**
1. Current Architecture: [Architecture v1](reference/architecture.md)
2. Roadmap: [Extended Models Roadmap](roadmap-extended-models.md)

---

## 🔄 Version History

| Version | Status | Description |
|---------|--------|-------------|
| **v1** | ✅ Current | Mock mode - generates credible artifacts |
| **v2** | 🚧 In Progress | Trainable mode - generates working models |

See [Extended Models Roadmap](roadmap-extended-models.md) for progress.

---

## Documentation Conventions

- Paths are relative to `docs/`
- Example specs live under `examples/specs/`

---

**Last Updated**: 2025-12-28
**Version**: 2.0 (reorganized for v2 development)
