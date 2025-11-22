# Lightweight Dockerfile for MARL development
# Uses Ubuntu 22.04 base (~2GB) - works on systems with or without GPU
# PyTorch automatically uses GPU if available via Docker GPU passthrough

FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONPATH=/workspace:/workspace/vendor/multigrid

# Set working directory
WORKDIR /workspace

# Install system dependencies (no CUDA libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    build-essential \
    libopenmpi-dev \
    openmpi-bin \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create python symlink and upgrade pip
RUN ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt

# Install Python dependencies
# Use CPU-only PyTorch to avoid downloading large CUDA packages (~2GB vs ~5GB)
# GPU will still work if available via Docker GPU passthrough (--gpus flag)
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple \
    -r /tmp/requirements.txt

# Install dev dependencies only if DEV_MODE is set (for Docker Compose)
# This allows the same Dockerfile to be used for both dev and production
ARG DEV_MODE=false
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    if [ "$DEV_MODE" = "true" ] ; then \
    pip install -r /tmp/requirements-dev.txt ; \
    fi

# Create a non-root user for better security
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd -g ${GROUP_ID} appuser && \
    useradd -m -u ${USER_ID} -g appuser -s /bin/bash appuser

# Create workspace directory and set permissions
RUN mkdir -p /workspace && chown -R appuser:appuser /workspace

# Switch to non-root user
USER appuser

# Create common output directories
RUN mkdir -p /workspace/outputs /workspace/logs

# Default command (can be overridden)
CMD ["/bin/bash"]
