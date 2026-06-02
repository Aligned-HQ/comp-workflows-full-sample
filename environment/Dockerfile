FROM python:3.12-slim

ENV WORKSPACE_DIR=/workspace \
    DATA_DIR=/workspace/data \
    OUTPUT_DIR=/workspace/output \
    PIPELINE_PATH=/workspace/output/pipeline.py \
    SOLUTION_DIR=/solution \
    TESTS_DIR=/tests \
    LOG_DIR=/logs/verifier

# Python packages required by the task and verifier.
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    pandas==2.2.2 \
    scipy==1.14.0 \
    scikit-learn==1.5.0 \
    GEOparse==2.0.4 \
    pytest==8.4.1 \
    pytest-json-ctrf==0.3.5

WORKDIR ${WORKSPACE_DIR}

# Copy gene-expression series data shipped with the task.
RUN mkdir -p ${DATA_DIR}
COPY data/GSE6269_family.soft.gz ${DATA_DIR}/GSE6269_family.soft.gz

# Runtime user expected by task.toml, plus writable output directory.
RUN useradd --create-home --shell /bin/bash agent \
    && mkdir -p ${OUTPUT_DIR} \
    && chown -R agent:agent ${WORKSPACE_DIR}
