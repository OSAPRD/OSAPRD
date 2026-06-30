FROM python:3.12-trixie

WORKDIR /workspace

# Git is required for optional wiki cloning and for installing Spiral from its
# upstream repository.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY topic-classification/requirements.txt /tmp/topic-classification-requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/topic-classification-requirements.txt

# NLTK corpora are installed at image build time so local runs do not download
# data during preprocessing.
ENV NLTK_DATA=/usr/local/share/nltk_data
RUN python -m nltk.downloader -d "$NLTK_DATA" stopwords wordnet omw-1.4

RUN mkdir -p /data/curation-outputs /data/runs /model /wiki-cache

# Defaults make the image runnable for classification with mounted curation
# outputs and a mounted model bundle. Extraction, preparation, and training use
# the same entrypoint with a different stage argument.
ENV POST_PROCESSING_CURATION_OUTPUTS_DIR=/data/curation-outputs
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_RUNS_DIR=/data/runs
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_MODEL_BUNDLE_PATH=/model/topic_model_bundle.joblib
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_WIKI_CACHE_DIR=/wiki-cache
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_WORKERS=1
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_INPUT_FORMAT=parquet
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_ENABLE_LIVE_WIKI_FETCH=true
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_REQUIRE_MODEL_PREPROCESSING_ARTIFACTS=true
# The README training flow works without optional raw-frequency artifacts. Set
# this to true only when training used --require-raw-frequency-artifacts with
# source-specific token-count files.
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_REQUIRE_RAW_FREQUENCY_ARTIFACTS=false
ENV POST_PROCESSING_TOPIC_CLASSIFICATION_IDENTIFIER_SPLITTER=spiral

COPY . /workspace/post-processing
RUN chmod -R a+rX /workspace/post-processing

CMD ["python", "-u", "post-processing/topic-classification/run.py", "classify"]
