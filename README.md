# Richter Predictor - Advanced Earthquake Damage Classification

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.18](https://img.shields.io/badge/TensorFlow-2.18-orange.svg)](https://tensorflow.org/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://docker.com/)
[![F1-Score](https://img.shields.io/badge/F1--Score-0.7055-green.svg)](models/)

Sistema avanzato di machine learning per la classificazione del danno sismico agli edifici basato sui dati del terremoto di Gorkha 2015 in Nepal. Il progetto implementa architetture ensemble ottimizzate per massimizzare l'F1-score attraverso feature engineering avanzato e validazione rigorosa anti-leakage.

## Indice

- [Struttura del Progetto](#struttura-del-progetto)
- [Architettura e Design](#architettura-e-design)
- [Setup e Installazione](#setup-e-installazione)
- [Configurazione CUDA/WSL](#configurazione-cudawsl)
- [Utilizzo](#utilizzo)
- [Modelli e Performance](#modelli-e-performance)
- [Testing](#testing)
- [Deployment](#deployment)

## Struttura del Progetto

```
richter-predictor-fia/
├── src/                              # Codice sorgente principale
│   ├── data/                         # Moduli per analisi e EDA
│   │   ├── data_analysis.py          # Analisi dei dati core
│   │   └── eda.py                    # Exploratory Data Analysis
│   ├── feature_engineering/          # Feature engineering avanzato
│   │   ├── __init__.py
│   │   └── advanced_features.py      # Features domain-specific
│   ├── preprocessing/                # Pipeline di preprocessing modulare
│   │   ├── __init__.py
│   │   ├── base_preprocessor.py      # Classe base astratta
│   │   ├── binary_preprocessor.py    # Features binarie
│   │   ├── categorical_preprocessor.py # Features categoriche + embeddings
│   │   ├── geographic_preprocessor.py # Features geografiche
│   │   ├── numeric_preprocessor.py   # Features numeriche + scaling
│   │   └── main_pipeline.py          # Pipeline integrata TensorFlow
│   └── models/                       # Modelli e training
│       ├── ensemble_architectures.py # 6 architetture diverse per ensemble
│       ├── leakage_validator.py      # Validazione anti-leakage
│       ├── train_advanced_ensemble.py # Training ensemble principale
│       ├── train_nested_cv_ensemble.py # Training con Nested CV
│       └── train_simple_holdout.py   # Training semplice per debug
├── data/                             # Dataset
│   └── raw/
│       ├── train_values.csv          # Features di training
│       ├── train_labels.csv          # Target di training
│       └── test_values.csv           # Features per submission
├── models/                           # Modelli salvati
│   ├── ensemble_f1_0.7055_20250726_185224/ # Modello migliore attuale
│   │   ├── config.json               # Configurazione training
│   │   ├── feature_engineer.pkl      # Feature engineering fitted
│   │   ├── preprocessing_pipeline.pkl # Pipeline preprocessing
│   │   └── model_*.keras             # 6 modelli ensemble
│   └── ...                          # Altri modelli salvati
├── reports/                          # Report e analisi
│   ├── eda/                          # Risultati EDA
│   │   ├── figures/                  # Grafici e visualizzazioni
│   │   ├── tables/                   # Tabelle analitiche
│   │   └── *.json, *.csv             # Dati strutturati
│   └── mlp_results/                  # Risultati modelli precedenti
├── tests/                            # Test suite completa
│   ├── run_tests.py                  # Test runner consolidato
│   ├── test_core.py                  # Test preprocessing
│   ├── test_models.py                # Test modelli ML
│   ├── test_ensemble.py              # Test ensemble
│   ├── test_integration.py           # Test integrazione
│   └── test_data/                    # Dati sintetici per test
├── venv/                             # Virtual environment Python
├── logs/                             # Log di training e debug
├── Dockerfile                        # Container per deployment
├── docker-compose.yml                # Orchestrazione multi-servizio
├── docker-helper.sh                  # Script helper Docker
├── requirements.txt                  # Dipendenze Python
├── debug_sparsity.py                 # Debug sparsity dati
├── test_nested_cv_subset.py          # Test Nested CV su subset
└── README.md                         # Questa documentazione
```

## Architettura e Design

### Design Pattern Utilizzati

#### 1. **Modular Preprocessing Pipeline**
- **Pattern**: Strategy + Template Method
- **Motivazione**: Gestire diversi tipi di features (numeriche, categoriche, geografiche, binarie) con preprocessing specializzato
- **Implementazione**: Ogni preprocessore eredita da `PreprocessorPipeline` e implementa metodi specifici
- **Benefici**: Estendibilità, testabilità, riusabilità

#### 2. **Ensemble Learning con Diversità Architettuale**
- **Pattern**: Ensemble Methods + Factory
- **Motivazione**: Massimizzare performance combinando modelli con architetture diverse
- **Implementazione**: 6 architetture distinte (deep narrow, wide shallow, residual-like, regularized, swish activation, attention-like)
- **Benefici**: Riduce overfitting, migliora generalizzazione

#### 3. **Feature Engineering Domain-Driven**
- **Pattern**: Domain-Driven Design
- **Motivazione**: Sfruttare knowledge domain sismico per creare features meaningful
- **Implementazione**: Features geografiche intelligenti, risk scores, interaction terms
- **Benefici**: Migliore performance predittiva, interpretabilità

#### 4. **Anti-Leakage Validation**
- **Pattern**: Cross-Validation con Nested CV
- **Motivazione**: Evitare data leakage nel feature engineering e hyperparameter tuning
- **Implementazione**: Nested Cross-Validation con validazione separata
- **Benefici**: Stime realistiche delle performance, robustezza

### Scelte Tecnologiche

#### **TensorFlow 2.18 con Keras**
- **Motivazione**: Ecosystem maturo, ottimizzazione GPU, facilità deployment
- **Benefici**: Preprocessing integrato nel grafo, serializzazione modelli, scalabilità

#### **Scikit-learn per Preprocessing**
- **Motivazione**: API standardizzata, preprocessori robusti e testati
- **Benefici**: Interoperabilità, documentazione estesa, community support

#### **Docker + Docker Compose**
- **Motivazione**: Reproducibilità, isolamento dipendenze, deployment semplificato
- **Benefici**: Environment consistency, CI/CD ready, scalabilità orizzontale

## Setup e Installazione

### Prerequisiti

- **Sistema Operativo**: Linux (Ubuntu 20.04+), macOS, Windows 10/11 con WSL2
- **Python**: 3.12+
- **RAM**: Minimo 8GB, raccomandato 16GB+
- **Storage**: Minimo 5GB liberi
- **GPU** (opzionale): NVIDIA GPU con CUDA 12.x per accelerazione

### Installazione Locale

#### 1. Clona il Repository
```bash
git clone https://github.com/DanieleLimongi/richter-predictor-fia.git
cd richter-predictor-fia
```

#### 2. Crea e Attiva Virtual Environment
```bash
# Crea virtual environment
python3.12 -m venv venv

# Attiva virtual environment
# Su Linux/macOS:
source venv/bin/activate

# Su Windows (WSL):
source venv/bin/activate

# Su Windows (Command Prompt):
venv\Scripts\activate
```

#### 3. Aggiorna pip e Installa Dipendenze
```bash
# Aggiorna pip
pip install --upgrade pip

# Installa dipendenze
pip install -r requirements.txt

# Verifica installazione TensorFlow
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed successfully')"
```

#### 4. Setup Variabili Ambiente
```bash
# Aggiungi al tuo .bashrc o .zshrc
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src:$(pwd)/tests"
export TF_CPP_MIN_LOG_LEVEL=2  # Riduce verbosity TensorFlow
```

#### 5. Verifica Setup
```bash
# Test rapido del setup
python tests/run_tests.py --test utils

# Test completo (opzionale, richiede più tempo)
python tests/run_tests.py
```

## Configurazione CUDA/WSL

### Setup CUDA su WSL2 (Windows)

#### 1. Installa WSL2 e Ubuntu
```powershell
# Da PowerShell come Administrator
wsl --install -d Ubuntu-22.04
wsl --set-version Ubuntu-22.04 2
```

#### 2. Installa NVIDIA WSL Driver
- Scarica da: https://developer.nvidia.com/cuda/wsl
- Installa il driver WSL-specific (NON il driver standard)

#### 3. Setup CUDA in WSL
```bash
# Entra in WSL
wsl

# Aggiorna sistema
sudo apt update && sudo apt upgrade -y

# Installa CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-6

# Aggiungi CUDA al PATH
echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

#### 4. Verifica CUDA
```bash
# Verifica CUDA
nvcc --version
nvidia-smi

# Test TensorFlow con GPU
python -c "
import tensorflow as tf
print('GPU Available:', len(tf.config.list_physical_devices('GPU')) > 0)
print('GPU Devices:', tf.config.list_physical_devices('GPU'))
"
```

### Setup CUDA su Linux Nativo

#### 1. Installa Driver NVIDIA
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install nvidia-driver-535  # O versione più recente

# Riavvia sistema
sudo reboot
```

#### 2. Installa CUDA Toolkit
```bash
# Scarica e installa CUDA 12.6
wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda_12.6.0_560.28.03_linux.run
sudo sh cuda_12.6.0_560.28.03_linux.run

# Segui le istruzioni, NON installare il driver se già presente
```

#### 3. Configura Environment
```bash
# Aggiungi a ~/.bashrc
echo 'export PATH=/usr/local/cuda-12.6/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

## Utilizzo

### Metodo 1: Docker (Raccomandato)

#### Setup Iniziale
```bash
# Rendi eseguibile lo script helper
chmod +x docker-helper.sh

# Setup completo (build + directory)
./docker-helper.sh setup
```

#### Comandi Principali
```bash
# Training ensemble avanzato
./docker-helper.sh train

# Training con Nested CV (anti-leakage)
./docker-helper.sh train-nested

# Training semplice per debug
./docker-helper.sh train-simple

# Test suite completa
./docker-helper.sh test

# EDA e analisi dati
./docker-helper.sh eda

# Shell interattiva
./docker-helper.sh shell

# Visualizza tutti i comandi
./docker-helper.sh help
```

### Metodo 2: Locale

#### Training
```bash
# Attiva virtual environment
source venv/bin/activate

# Training ensemble avanzato
python src/models/train_advanced_ensemble.py

# Training con Nested CV
python src/models/train_nested_cv_ensemble.py

# Training semplice
python src/models/train_simple_holdout.py
```

#### Analisi Dati
```bash
# EDA completa
python src/data/eda.py --raw_dir data/raw --output_dir reports/eda

# Analisi dati
python src/data/data_analysis.py

# Debug sparsity
python debug_sparsity.py
```

#### Testing
```bash
# Test completi
python tests/run_tests.py

# Test specifici
python tests/run_tests.py --test preprocessing
python tests/run_tests.py --test models
python tests/run_tests.py --test utils

# Test Nested CV su subset
python test_nested_cv_subset.py
```

## Modelli e Performance

### Architetture Ensemble

Il sistema implementa 6 architetture diverse per massimizzare la diversità:

1. **Deep Narrow**: Rete profonda con layer stretti per pattern complessi
2. **Wide Shallow**: Rete larga con pochi layer per pattern semplici  
3. **Residual-like**: Architettura con connessioni skip
4. **Regularized**: Heavy regularization (Dropout + L2)
5. **Swish Activation**: Activation function Swish invece di ReLU
6. **Attention-like**: Meccanismo pseudo-attention

### Performance Attuali

| Modello | F1-Score | Data Training | Note |
|---------|----------|---------------|------|
| **ensemble_f1_0.7055** | **0.7055** | 2025-07-26 | Migliore attuale |
| ensemble_f1_0.6809 | 0.6809 | 2025-07-26 | Versione precedente |
| mlp_model_full | 0.6500 | 2025-07-23 | Baseline MLP |

### Feature Engineering

#### Features Domain-Specific
- **Geographic Risk**: Encoding intelligente delle zone geografiche
- **Material Combinations**: Interazioni tra materiali strutturali
- **Vulnerability Scores**: Score di vulnerabilità basato su domain knowledge
- **Building Age Interactions**: Interazioni età-materiali
- **Polynomial Features**: Features polinomiali per relazioni non-lineari

#### Anti-Leakage Validation
- **Nested Cross-Validation**: Evita leakage nel feature selection
- **Temporal Validation**: Rispetta ordine temporale dei dati
- **Independent Test Set**: Validazione su dati completamente separati

## Testing

### Test Suite Strutturata

```bash
# Test completi (tutti i moduli)
python tests/run_tests.py

# Test per categoria
python tests/run_tests.py --test preprocessing  # Pipeline preprocessing
python tests/run_tests.py --test models        # Modelli ML
python tests/run_tests.py --test utils         # Utilities
python tests/run_tests.py --test ensemble      # Sistema ensemble

# Test CI/CD
python tests/run_tests.py --ci

# Test rapidi
python tests/run_tests.py --test utils --quiet
```

### Coverage Testing

| Modulo | Coverage | Descrizione |
|--------|----------|-------------|
| **Preprocessing** | 95%+ | Pipeline completa, edge cases |
| **Models** | 90%+ | Architetture, training, inference |
| **Feature Engineering** | 85%+ | Transformation, validation |
| **Integration** | 80%+ | End-to-end workflows |

## Deployment

### Docker Production

#### Build Ottimizzato
```bash
# Build immagine production
docker build -t richter-predictor:production .

# Run container
docker run -d \
  --name richter-prod \
  -v $(pwd)/data:/app/data:ro \
  -v $(pwd)/models:/app/models:ro \
  -p 8000:8000 \
  richter-predictor:production
```

#### Docker Compose Services

```bash
# Servizio training
docker-compose --profile training up

# Servizio testing
docker-compose --profile testing up

# Servizio submission
docker-compose --profile submission up
```

### Monitoring e Logging

- **Logs**: Centralizzati in `logs/` directory
- **Metrics**: TensorBoard logs per training monitoring
- **Health Checks**: Endpoint Docker per monitoring container
- **Performance**: Tracking F1-score e training metrics

## Troubleshooting

### Problemi Comuni

#### GPU Non Rilevata
```bash
# Verifica driver NVIDIA
nvidia-smi

# Test TensorFlow GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# Forza CPU se GPU problematica
export CUDA_VISIBLE_DEVICES=""
```

#### Memory Issues
```bash
# Riduce batch size nel training
export RICHTER_BATCH_SIZE=32

# Limita memoria GPU TensorFlow
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### Dipendenze Mancanti
```bash
# Re-installa requirements
pip install --force-reinstall -r requirements.txt

# Controlla versioni specifiche
pip list | grep -E "(tensorflow|numpy|pandas|scikit-learn)"
```

### Debug Avanzato

```bash
# Debug sparsity dati
python debug_sparsity.py

# Test Nested CV su subset piccolo
python test_nested_cv_subset.py

# Validazione anti-leakage
python src/models/leakage_validator.py
```

## Roadmap

### Sviluppi Futuri

- [ ] **AutoML Integration**: Hyperparameter tuning automatico
- [ ] **MLOps Pipeline**: CI/CD completo per ML
- [ ] **Model Serving**: API REST per inference real-time
- [ ] **Feature Store**: Sistema centralizzato per feature management
- [ ] **A/B Testing**: Framework per testing modelli in produzione

### Ottimizzazioni

- [ ] **Quantization**: Riduzione dimensione modelli
- [ ] **Pruning**: Rimozione neuroni ridondanti
- [ ] **Knowledge Distillation**: Trasferimento knowledge a modelli più piccoli
- [ ] **Edge Deployment**: Ottimizzazione per dispositivi edge

## Contributi

1. Fork del repository
2. Crea branch feature (`git checkout -b feature/amazing-feature`)
3. Commit delle modifiche (`git commit -m 'Add amazing feature'`)
4. Push del branch (`git push origin feature/amazing-feature`)
5. Apri Pull Request

## Licenza

Progetto sviluppato per competizione Kaggle - Gorkha Earthquake Damage Classification.

---

**Sviluppato per il Nepal Earthquake Recovery**

*Per domande o supporto, apri un issue nel repository.*