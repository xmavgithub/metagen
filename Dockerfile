# MetaGen Docker Image
# Multi-stage build for smaller final image

# Stage 1: Build
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN pip install --no-cache-dir build

# Copy source
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Build wheel
RUN python -m build --wheel

# Stage 2: Runtime
FROM python:3.11-slim

LABEL maintainer="MetaGen Contributors"
LABEL description="MetaGen: Spec-to-Model Synthesizer"
LABEL version="1.0.0"

WORKDIR /app

# Install LaTeX (minimal for paper generation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    texlive-latex-base \
    texlive-latex-extra \
    texlive-fonts-recommended \
    texlive-science \
    latexmk \
    && rm -rf /var/lib/apt/lists/*

# Copy wheel from builder
COPY --from=builder /app/dist/*.whl /tmp/

# Install MetaGen
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy example specs
COPY examples/ examples/

# Create output directory
RUN mkdir -p /app/outputs

# Set default working directory for outputs
VOLUME ["/app/outputs"]

# Default command shows help
ENTRYPOINT ["metagen"]
CMD ["--help"]
