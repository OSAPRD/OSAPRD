FROM python:3.12-trixie

WORKDIR /workspace

# Only CA certificates are needed at the OS level; all processing tools are
# Python libraries installed from the stage requirements file.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY post-processing/upload-extraction-data/requirements.txt /tmp/upload-extraction-requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/upload-extraction-requirements.txt

# Runtime contract:
# - /data/input is a read-only extraction output mount.
# - /data/output is the writable staging tree and upload cache.
RUN mkdir -p /huggingfaceCache /data/input /data/output

ENV HF_HOME=/huggingfaceCache
ENV POST_PROCESSING_UPLOAD_EXTRACTION_SOURCE_DIRS=/data/input
ENV POST_PROCESSING_UPLOAD_EXTRACTION_OUTPUT_DIR=/data/output

COPY . .

CMD ["python", "-u", "post-processing/upload-extraction-data/run.py", "all"]
