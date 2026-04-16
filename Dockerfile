# Default (CPU) image — PyTorch CPU wheels for a smaller image.
# Build: docker build -t multi-uav-grid:latest .
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore

WORKDIR /app

COPY pyproject.toml README.md ./
COPY multi_uav_grid ./multi_uav_grid

RUN pip install --upgrade pip \
    && pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "numpy>=1.24" \
    && pip install --no-cache-dir --no-deps -e .

CMD ["python", "-m", "multi_uav_grid"]
