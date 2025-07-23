# Dockerfile per Richter Predictor
FROM python:3.12-slim

# Metadata
LABEL maintainer="claudio@richter-predictor.com"
LABEL description="Richter Predictor - Earthquake Damage Classification"
LABEL version="1.0"

# Variabili di ambiente
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV CUDA_VISIBLE_DEVICES=""

# Aggiorna sistema e installa dipendenze di sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Crea utente non-root per sicurezza
RUN useradd -m -u 1000 richter && \
    mkdir -p /app && \
    chown richter:richter /app

# Imposta working directory
WORKDIR /app

# Copia requirements prima per sfruttare Docker cache
COPY requirements.txt .

# Installa dipendenze Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice
COPY . .

# Imposta permessi corretti
RUN chown -R richter:richter /app

# Switcha all'utente non-root
USER richter

# Crea directories necessarie
RUN mkdir -p data/raw data/interim models reports logs tests

# Verifica che i test siano presenti
RUN ls -la tests/ || echo "Tests directory not found"

# Esponi porta per eventuali servizi (opzionale)
EXPOSE 8000

# Health check incluso test di base
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import tensorflow as tf; import sys; sys.path.append('tests'); print('TensorFlow OK')" && \
        python tests/run_tests.py --ci --test utils || exit 1

# Default command con opzioni test integrate
CMD ["python", "-c", "print(' Richter Predictor Docker Container Ready!'); print(' Available commands:'); print('  • python create_submission_fixed.py                    # Create submission'); print('  • python src/models/train_final_nested_cv.py          # Train with Nested CV'); print('  • python tests/run_tests.py                          # Run full test suite'); print('  • python tests/run_tests.py --test preprocessing     # Test preprocessing only'); print('  • python tests/run_tests.py --test models            # Test models only'); print('  • python tests/run_tests.py --test utils             # Test utilities only'); print('  • python tests/run_tests.py --ci                     # CI mode for automation'); print('  • python docker-helper.sh                            # Helper commands')"]
