FROM python:3.12.7

# Build context is the extraction/ directory. The source is copied to
# /extraction, while PYTHONPATH exposes / so `python -m extraction.run` can
# resolve the package in the same way as local repository-root execution.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/ \
    LOCAL_OUTPUT_DIR=/data

WORKDIR /extraction

# Install dependencies before copying source so Docker can reuse the dependency
# layer when only extraction code changes.
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /data

COPY . .

CMD ["python", "-m", "extraction.run"]
