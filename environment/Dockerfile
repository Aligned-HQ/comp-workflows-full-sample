FROM python:3.12-slim

# Python packages required by the task itself.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.14.0 \
    scikit-learn==1.5.0 \
    GEOparse==2.0.4

# Set working directory
WORKDIR /workspace

# Copy gene-expression series data shipped with the task.
COPY data/GSE6269_family.soft.gz /workspace/data/GSE6269_family.soft.gz

# Runtime user expected by task.toml, plus writable output directory.
RUN useradd --create-home --shell /bin/bash agent \
    && mkdir -p /workspace/output \
    && chown -R agent:agent /workspace
