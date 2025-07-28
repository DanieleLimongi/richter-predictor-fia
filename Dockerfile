# Richter Predictor FIA - Optimized Production Dockerfile
FROM python:3.12-slim

# Metadata
LABEL maintainer="danielelimongi@richter-predictor.com"
LABEL description="Richter Predictor FIA - Seismic Damage Classification"
LABEL version="2.0.0"

# Build arguments
ARG DEBIAN_FRONTEND=noninteractive

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Create application user
RUN groupadd -r richter --gid=1000 && \
    useradd -r -g richter --uid=1000 --home-dir=/app --shell=/bin/bash richter && \
    mkdir -p /app && \
    chown -R richter:richter /app

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip cache purge && \
    rm -rf /tmp/requirements.txt

# Verify critical dependencies
RUN python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')" && \
    python -c "import pandas, numpy, sklearn; print('Core dependencies OK')"

# Switch to non-root user
USER richter

# Create directory structure
RUN mkdir -p \
    data/raw data/interim \
    models reports logs submissions tests/test_data \
    .cache

# Copy application code
COPY --chown=richter:richter src/ /app/src/
COPY --chown=richter:richter tests/ /app/tests/

# Set correct permissions
RUN find /app -type f -name "*.py" -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=2 \
    CMD python -c "import tensorflow as tf; import sys; sys.path.append('/app/src'); from feature_engineering import AdvancedFeatureEngineer; print('OK')" || exit 1

# Expose port for future web services
EXPOSE 8000

# Default command
CMD ["tail", "-f", "/dev/null"]