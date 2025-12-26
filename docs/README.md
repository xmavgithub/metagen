# MetaGen Documentation

**Welcome to MetaGen documentation!**

MetaGen is a spec-to-model synthesizer that generates complete AI model release artifacts from high-level YAML specifications.

---

## 📚 Documentation Index

### For Users

**Getting Started**
- [Spec Language (MSL)](user-guide/spec_language.md) - How to write model specifications
- [FAQ](user-guide/faq.md) - Frequently asked questions

**Usage Guides**
- [Paper Generation](user-guide/paper.md) - Generate LaTeX papers
- [Benchmarks](user-guide/benchmarks.md) - Understanding benchmark results

---

### For Developers

**Project Planning**
- [Roadmap](project/roadmap.md) - Development roadmap and work units ⭐
- [Original Specification](project/meta_gen_coding_agent_specification.md) - Initial project requirements

**Architecture & Design**
- [Architecture v1](reference/architecture.md) - Current architecture (mock mode)
- [Architecture v2](development/architecture_v2.md) - BlueprintState system design (trainable mode) ⭐
- [Trainable Models Spec](development/trainable_models_spec.md) - Technical specifications for trainable models ⭐

**Development Practices**
- [Testing Strategy](development/testing_strategy.md) - Testing levels, markers, and best practices ⭐

---

## 🎯 Quick Navigation

### Current State (v1 - Mock Mode)
MetaGen currently generates **credible release artifacts** without actual training:
- Read: [Architecture v1](reference/architecture.md)
- Read: [Benchmarks](user-guide/benchmarks.md)
- Read: [FAQ](user-guide/faq.md)

### Future State (v2 - Trainable Mode)
We're transforming MetaGen to generate **actually trainable models**:
- **Start here**: [Roadmap](project/roadmap.md) - See the 8 work units
- Architecture: [BlueprintState Design](development/architecture_v2.md)
- Specs: [Trainable Models](development/trainable_models_spec.md)
- Testing: [Testing Strategy](development/testing_strategy.md)

---

## 📂 Folder Structure

```
docs/
├── README.md                          # This file
├── user-guide/                        # For end users
│   ├── spec_language.md              # How to write specs
│   ├── faq.md                        # Frequently asked questions
│   ├── paper.md                      # Paper generation guide
│   └── benchmarks.md                 # Benchmark interpretation
├── development/                       # For contributors
│   ├── trainable_models_spec.md      # Technical spec for trainable models
│   ├── architecture_v2.md            # BlueprintState system design
│   └── testing_strategy.md           # Testing guidelines
├── reference/                         # Reference documentation
│   └── architecture.md               # Current architecture (v1)
└── project/                          # Project management
    ├── roadmap.md                    # Development roadmap
    └── meta_gen_coding_agent_specification.md  # Original spec
```

---

## 🚀 Key Documents by Role

### **I'm a User**
1. Start: [Spec Language](user-guide/spec_language.md)
2. Questions: [FAQ](user-guide/faq.md)
3. Results: [Benchmarks](user-guide/benchmarks.md)

### **I'm a Contributor**
1. Start: [Roadmap](project/roadmap.md) - See what's being built
2. Architecture: [Architecture v2](development/architecture_v2.md) - Understand the design
3. Testing: [Testing Strategy](development/testing_strategy.md) - Write good tests
4. Specs: [Trainable Models](development/trainable_models_spec.md) - Implementation details

### **I'm Curious About the Project**
1. Vision: [Original Specification](project/meta_gen_coding_agent_specification.md)
2. Current: [Architecture v1](reference/architecture.md)
3. Future: [Roadmap](project/roadmap.md)

---

## 🔄 Version History

| Version | Status | Description |
|---------|--------|-------------|
| **v1** | ✅ Current | Mock mode - generates credible artifacts |
| **v2** | 🚧 In Progress | Trainable mode - generates working models |

See [Roadmap](project/roadmap.md) for v2 progress.

---

## 📖 Documentation Conventions

- ⭐ = Key document for v2 development
- 🚧 = Work in progress
- ✅ = Stable/complete
- 📝 = Needs update

---

**Last Updated**: 2025-12-23
**Version**: 2.0 (reorganized for v2 development)
