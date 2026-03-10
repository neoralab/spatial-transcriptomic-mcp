FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1 \
    PORT=8080

WORKDIR /app

# Build dependencies for Python packages with native extensions (for example `annoy`)
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install ChatSpatial package from source with uv
COPY . /app
RUN pip install --no-cache-dir --upgrade pip uv \
    && uv pip install --system --no-cache .

EXPOSE 8080

# Cloud Run requires binding to 0.0.0.0:$PORT
CMD ["sh", "-c", "python -m chatspatial server --transport ${CHATSPATIAL_TRANSPORT:-streamable-http} --host ${CHATSPATIAL_HOST:-0.0.0.0} --port ${PORT:-8080} --log-level ${CHATSPATIAL_LOG_LEVEL:-INFO}"]
