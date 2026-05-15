FROM python:3.12-slim

# Install Python packages. `uv` provides the `uvx` command used by
# tests/test.sh to spin up an isolated pytest environment.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.14.0 \
    scikit-learn==1.5.0 \
    GEOparse==2.0.4 \
    uv==0.5.11

# Set working directory
WORKDIR /workspace

# Copy gene-expression series data shipped with the task.
COPY data/GSE6269_family.soft.gz /workspace/data/GSE6269_family.soft.gz

# Create output directory
RUN mkdir -p /workspace/output
