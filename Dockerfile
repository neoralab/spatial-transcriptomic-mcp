FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Install ChatSpatial package from source
COPY . /app
RUN pip install --upgrade pip && pip install .

EXPOSE 8080

# Cloud Run requires binding to 0.0.0.0:$PORT
CMD ["sh", "-c", "python -m chatspatial server --cloud-run --log-level INFO"]
