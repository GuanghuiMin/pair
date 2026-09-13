# OfficeBench runtime for pair.benchmarks.officebench: Python 3.12, LibreOffice and Tesseract for the
# office apps, the pinned OfficeBench dependencies, and tiktoken with its encodings pre-downloaded so
# a container only ever reaches the network for the LLM endpoint.
#
#     docker build -t pair-officebench docker/officebench
#
# The ACON checkout, this repository and the canonical task export are bind-mounted at run time.
FROM python:3.12-slim-bookworm
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      libreoffice-calc libreoffice-writer libreoffice-impress libreoffice-core \
      tesseract-ocr tesseract-ocr-eng imagemagick git tree jq curl ca-certificates \
      fonts-dejavu fonts-liberation procps && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r /tmp/requirements.txt tiktoken==0.9.0 pytz
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken_cache
RUN mkdir -p /opt/tiktoken_cache && \
    python3 -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o'); tiktoken.get_encoding('cl100k_base')"
WORKDIR /work
