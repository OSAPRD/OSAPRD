FROM python:3.12-trixie

WORKDIR /workspace

# The uploader only needs TLS certificates at the OS layer. Parquet handling and
# Hugging Face access are installed from the stage-scoped Python requirements.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY post-processing/upload-curation-data/requirements.txt /tmp/upload-curation-requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/upload-curation-requirements.txt

# Runtime contract:
# - /data/input is a read-only curation output mount.
# - /data/topic-classification optionally contains topic-classification outputs.
# - /data/longitudinal-refactoring optionally contains longitudinal summaries.
# - /data/output is the writable staging tree and upload cache.
RUN mkdir -p \
    /huggingfaceCache \
    /data/input \
    /data/topic-classification \
    /data/longitudinal-refactoring \
    /data/output

ENV HF_HOME=/huggingfaceCache
ENV POST_PROCESSING_UPLOAD_CURATION_INPUT_DIR=/data/input
ENV POST_PROCESSING_UPLOAD_CURATION_TOPIC_CLASSIFICATION_DIR=/data/topic-classification
ENV POST_PROCESSING_UPLOAD_CURATION_LONGITUDINAL_REFACTORING_DIR=/data/longitudinal-refactoring
ENV POST_PROCESSING_UPLOAD_CURATION_OUTPUT_DIR=/data/output

# Copy the repository after dependency installation so Docker can reuse the
# dependency layer when only source files change.
COPY . .

CMD ["python", "-u", "post-processing/upload-curation-data/run.py", "all"]
