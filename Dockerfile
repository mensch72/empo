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
# Use BuildKit cache mount for apt to avoid re-downloading packages
# Add deadsnakes PPA for Python 3.11 (required by MineLand)
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
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
    graphviz

# Create python symlink to Python 3.11 and upgrade pip
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.11 /usr/bin/python && \
    python3.11 -m pip install --upgrade pip setuptools wheel

# Copy requirements files
COPY requirements.txt /tmp/requirements.txt
COPY requirements-dev.txt /tmp/requirements-dev.txt
COPY requirements-hierarchical.txt /tmp/requirements-hierarchical.txt

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

# Install hierarchical dependencies only if HIERARCHICAL_MODE is set
# MineLand requires Java JDK 17, Node.js 18.x, and xvfb for headless rendering
# MineLand is a multi-agent Minecraft RL platform from https://github.com/cocacola-lab/MineLand
ARG HIERARCHICAL_MODE=false
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    apt-get update && apt-get install -y --no-install-recommends \
        openjdk-17-jdk \
        xvfb \
        xauth \
        curl ; \
    fi
# Install Node.js 18.x for MineLand (using NodeSource repository)
RUN if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs ; \
    fi
RUN --mount=type=cache,target=/root/.cache/pip,uid=0,gid=0 \
    if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    pip install -r /tmp/requirements-hierarchical.txt ; \
    fi
# Clone and install MineLand from GitHub
# Note: Using main branch; consider pinning to a specific commit for reproducible builds
RUN if [ "$HIERARCHICAL_MODE" = "true" ] ; then \
    git clone --depth 1 https://github.com/cocacola-lab/MineLand.git /opt/MineLand && \
    cd /opt/MineLand && \
    pip install -e . && \
    cd /opt/MineLand/mineland/sim/mineflayer && \
    npm ci ; \
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
