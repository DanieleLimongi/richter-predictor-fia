# Richter Predictor FIA - Production Dockerfile
# Multi-stage build per ottimizzazione dimensioni e sicurezza

# ====================================================================
# STAGE 1: Base system con dipendenze di sistema
# ====================================================================
FROM python:3.12-slim as base

# Metadata del container
LABEL maintainer="danielelimongi@richter-predictor.com"
LABEL description="Richter Predictor FIA - Advanced Seismic Damage Classification"
LABEL version="2.0.0"
LABEL created="2025-07-27"
LABEL python.version="3.12"
LABEL tensorflow.version="2.18.0"

# Build arguments per flessibilità
ARG PYTHON_VERSION=3.12
ARG TF_VERSION=2.18.0
ARG DEBIAN_FRONTEND=noninteractive

# 🌍 Variabili di ambiente globali
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# 📦 Aggiorna sistema e installa dipendenze essenziali
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Build essentials
    build-essential \
    gcc \
    g++ \
    # Libraries per Python packages
    libhdf5-dev \
    libhdf5-serial-dev \
    pkg-config \
    # Utilities
    curl \
    wget \
    git \
    ca-certificates \
    # Per debugging (opzionale)
    htop \
    nano \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* /var/tmp/*

# ====================================================================
# 👤 STAGE 2: User setup e security
# ====================================================================
FROM base as security

# 🔒 Crea utente non-root per sicurezza
RUN groupadd -r richter --gid=1000 && \
    useradd -r -g richter --uid=1000 --home-dir=/app --shell=/bin/bash richter && \
    mkdir -p /app && \
    chown -R richter:richter /app

# 🏠 Imposta working directory
WORKDIR /app

# ====================================================================
# 🐍 STAGE 3: Python dependencies
# ====================================================================
FROM security as python-deps

# Copia requirements per layer caching ottimale
COPY requirements.txt /tmp/requirements.txt

# Upgrade pip e installa dipendenze Python
RUN pip install --no-cache-dir --upgrade pip==24.0 setuptools wheel && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    # Cleanup pip cache
    pip cache purge && \
    rm -rf /tmp/requirements.txt

# Verifica installazione dipendenze critiche
RUN python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')" && \
    python -c "import pandas, numpy, sklearn; print('Core dependencies OK')" && \
    python -c "import matplotlib, seaborn; print('Visualization libraries OK')"

# ====================================================================
# STAGE 4: Application code
# ====================================================================
FROM python-deps as app

# 👤 Switch a utente non-root
USER richter

# 📂 Crea struttura directory necessarie
RUN mkdir -p \
    data/raw \
    data/interim \
    models/single_models \
    models/ensemble_models \
    models/experiments \
    reports/eda/figures \
    reports/eda/tables \
    reports/model_performance \
    submissions \
    logs \
    tests/test_data \
    .cache

# Copia codice applicazione con permessi corretti
COPY --chown=richter:richter src/ /app/src/
COPY --chown=richter:richter tests/ /app/tests/
COPY --chown=richter:richter debug_sparsity.py /app/
COPY --chown=richter:richter test_nested_cv_subset.py /app/

# 🔒 Imposta permessi corretti per sicurezza
RUN find /app -type f -name "*.py" -exec chmod 644 {} \; && \
    find /app -type d -exec chmod 755 {} \;

# ====================================================================
# 🏥 STAGE 5: Health check e final setup
# ====================================================================
FROM app as final

# 🏥 Health check completo per monitoring
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD python -c "\
import sys, os; \
sys.path.append('/app/src'); \
try: \
    import tensorflow as tf; \
    import pandas as pd; \
    import numpy as np; \
    from feature_engineering import AdvancedFeatureEngineer; \
    print('All dependencies and modules healthy'); \
    exit(0); \
except Exception as e: \
    print(f'Health check failed: {e}'); \
    exit(1); \
"

# 🌐 Esponi porta per eventuali servizi web futuri
EXPOSE 8000

# Entry point intelligente con informazioni sistema
ENTRYPOINT ["python", "-c", "\
import sys, os, platform; \
print('RICHTER PREDICTOR FIA - CONTAINER READY'); \
print('='*60); \
print(f'Python: {platform.python_version()}'); \
print(f'Platform: {platform.platform()}'); \
print(f'Working Dir: {os.getcwd()}'); \
print(f'User: {os.getenv(\"USER\", \"richter\")}'); \
print('='*60); \
print('COMANDI DISPONIBILI:'); \
print('  Training:'); \
print('    python src/models/train_nested_cv_ensemble.py    # Nested CV ensemble'); \
print('    python src/models/train_simple_holdout.py        # Modello singolo'); \
print(''); \
print('  Testing:'); \
print('    python tests/run_tests.py                        # Test suite completa'); \
print('    python tests/run_tests.py --test core            # Test core components'); \
print('    python test_nested_cv_subset.py                  # Test CV rapido'); \
print(''); \
print('  Analisi:'); \
print('    python src/data/eda.py                           # EDA completa'); \
print('    python debug_sparsity.py                         # Debug sparsità'); \
print(''); \
print('  Submission:'); \
print('    python src/create_submission.py                  # Genera submission'); \
print(''); \
print('  Debugging:'); \
print('    ls models/                                        # Modelli disponibili'); \
print('    ls data/raw/                                      # Dataset raw'); \
print('    ls reports/                                       # Report generati'); \
print('='*60); \
print('🚀 Container ready! Usa docker exec per comandi interattivi.'); \
"]

# 🛌 Default command per container non-interattivo
CMD ["tail", "-f", "/dev/null"]
