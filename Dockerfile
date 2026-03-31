# syntax=docker/dockerfile:1

# Use Python 3.10 slim — required for mediapipe 0.10.x compatibility
FROM python:3.10-slim

WORKDIR /app

# ── System dependencies ─────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ─────────────────────────────────────────────────────
# Set pip timeout high to survive slow networks; retry 3 times
ENV PIP_DEFAULT_TIMEOUT=300 \
    PIP_RETRIES=3 \
    PIP_NO_CACHE_DIR=1

# Upgrade pip (use --timeout flag explicitly as extra safety)
RUN pip install --timeout 300 --upgrade pip setuptools wheel

# Install CPU-only PyTorch first in a separate layer (cacheable ~700MB step)
RUN pip install --timeout 300 \
    "torch>=2.0.0" \
    "torchvision>=0.15.0" \
    --index-url https://download.pytorch.org/whl/cpu

# Install the rest of the requirements
# Strip torch/torchvision/index-url lines (already installed above)
COPY requirements.txt .
RUN grep -vE "^torch|^torchvision|^--extra-index-url|^#|^$" requirements.txt \
    > /tmp/req_notorch.txt && \
    pip install --timeout 300 -r /tmp/req_notorch.txt

# ── Application files ────────────────────────────────────────────────────────
COPY . .

# ── Runtime ──────────────────────────────────────────────────────────────────
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=5 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
