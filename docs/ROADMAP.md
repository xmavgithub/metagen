# Project Roadmap

This document correlates with the `task.md` artifact and tracks the transformation of MetaGen into a trainable model generator.

## Work Units

### WU-1: Documentation & Infrastructure ✅
- Organized documentation (complete; site restructure is a separate initiative)
- Set up pytest markers
- Created initial roadmap

### WU-2: Blueprint Sync ✅
- **Goal**: Ensure architecture dimensions are consistent across the pipeline.
- **Changes**:
    - Introduced `BlueprintState` as single source of truth.
    - Updated `engine.py` and `codegen.py` to use blueprint dimensions.
    - Added integration tests for consistency.

### WU-3: Templates System ✅
- **Goal**: Replace hardcoded string concatenation with a robust templating engine (Jinja2).
- **Changes**:
    - Created `src/metagen/templates` with Jinja2 files.
    - Updated `codegen.py` to use `Environment.render()`.
    - Maintained full backward compatibility and determinism.

### WU-4: Data Loading ✅
- **Goal**: Implement real PyTorch DataLoaders for text and image datasets.
- **Changes**:
    - Implemented `SyntheticTextDataset` and `FileTextDataset` in `data.py.j2`.
    - Updated template rendering to pass blueprint context.
    - Verified generation validity with new tests.

### WU-5: Training Loop ✅
- **Goal**: Create a functional training loop (NanoGPT style) compatible with generated models.
- **Changes**:
    - Created `src/metagen/templates/train.py.j2` with device selection and AdamW.
    - Updated `model.py.j2` to include Embeddings (fixing execution on MPS).
    - Verified loop execution with integration tests.

### WU-6: Evaluation ✅
- **Goal**: Implement real loss calculation and basic metrics.
- **Changes**:
    - Created `src/metagen/templates/eval.py.j2` with loss and perplexity calculation.
    - Added standalone execution block for easy testing.
    - Verified with integration tests.

### WU-7: Integration ✅
- **Goal**: Tie everything together into a `train` command.
- **Changes**:
    - Added `metagen train <spec.yaml>` command to CLI.
    - Generates blueprint, code, and runs training via subprocess.

### WU-8: Release ✅
- **Goal**: Final polish and version bump.
- **Changes**:
    - Bumped version to `0.2.0`.
    - All Work Units complete!
