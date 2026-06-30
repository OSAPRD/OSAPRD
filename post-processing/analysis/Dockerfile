FROM python:3.12-trixie

WORKDIR /workspace

# Analysis only needs Python plotting/parquet runtime libraries and fontconfig.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    fontconfig \
    fonts-liberation2 \
    && rm -rf /var/lib/apt/lists/*

# Install the isolated analysis requirements instead of the full project stack.
COPY post-processing/analysis/requirements.txt /tmp/analysis-requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/analysis-requirements.txt

# Mounted input/output roots mirror the README examples.
RUN mkdir -p /data/curation-data /data/topic-classification /output

ENV POST_PROCESSING_ANALYSIS_CURATION_DATA_DIR=/data/curation-data
ENV POST_PROCESSING_ANALYSIS_TOPIC_CLASSIFICATION_OUTPUT_DIR=/data/topic-classification
ENV POST_PROCESSING_ANALYSIS_OUTPUT_DIR=/output
ENV POST_PROCESSING_ANALYSIS_PIPELINE=all
ENV POST_PROCESSING_ANALYSIS_MULTIMETRIC_SOURCE=auto

# Analysis imports only small taxonomy config modules from curation; it does not
# need curation tools, snapshots, or dependency caches.
COPY post-processing /workspace/post-processing
COPY curation/config /workspace/curation/config
COPY curation/__init__.py /workspace/curation/__init__.py
RUN chmod -R a+rX /workspace/post-processing /workspace/curation

CMD ["python", "-u", "post-processing/analysis/run.py", "all"]
