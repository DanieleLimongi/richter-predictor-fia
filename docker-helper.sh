#!/bin/bash

# Docker Utility Script per Richter Predictor
set -e

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}🎯 RICHTER PREDICTOR - DOCKER UTILITY${NC}"
echo "=========================================="

# Funzioni helper
log_info() {
    echo -e "${BLUE} $1${NC}"
}

log_success() {
    echo -e "${GREEN} $1${NC}"
}

log_warning() {
    echo -e "${YELLOW}  $1${NC}"
}

log_error() {
    echo -e "${RED} $1${NC}"
}

# Funzione per verificare Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        log_error "Docker non trovato! Installa Docker prima di continuare."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose non trovato! Installa Docker Compose prima di continuare."
        exit 1
    fi
    
    log_success "Docker e Docker Compose trovati!"
}

# Funzione per creare directory necessarie
setup_directories() {
    log_info "Creazione directory necessarie..."
    mkdir -p submissions logs
    log_success "Directory create!"
}

# Funzione per build dell'immagine
build_image() {
    log_info "Building Docker image..."
    docker-compose build richter-predictor
    log_success "Immagine creata con successo!"
}

# Funzione per training
run_training() {
    log_info "Avvio training del modello..."
    docker-compose run --rm richter-predictor python src/models/train_simple_holdout.py
    log_success "Training completato!"
}

# Funzione per submission - CORRETTA
run_submission() {
    log_info "Creazione submission..."
    docker-compose run --rm richter-predictor python create_submission.py
    log_success "Submission creata!"
}

# Funzione per shell interattiva
run_shell() {
    log_info "Avvio shell interattiva..."
    docker-compose run --rm richter-predictor bash
}

# Funzione per data analysis - SOSTITUTO DEI TEST
run_analysis() {
    log_info "Esecuzione analisi dati..."
    docker-compose run --rm richter-predictor python src/data/data_analysis.py
    log_success "Analisi completata!"
}

# Funzione per EDA
run_eda() {
    log_info "Generazione EDA e visualizzazioni..."
    docker-compose run --rm richter-predictor python src/data/eda.py --raw_dir data/raw --output_dir reports/eda
    log_success "EDA completata!"
}

# Funzioni per test suite
run_tests() {
    log_info "Esecuzione test suite completa..."
    docker-compose run --rm richter-predictor python tests/run_tests.py
    log_success "Test suite completata!"
}

run_tests_preprocessing() {
    log_info "Test preprocessing pipeline..."
    docker-compose run --rm richter-predictor python tests/run_tests.py --test preprocessing
    log_success "Test preprocessing completati!"
}

run_tests_models() {
    log_info "Test modelli ML..."
    docker-compose run --rm richter-predictor python tests/run_tests.py --test models
    log_success "Test modelli completati!"
}

run_tests_utils() {
    log_info "Test utilità..."
    docker-compose run --rm richter-predictor python tests/run_tests.py --test utils
    log_success "Test utilità completati!"
}

run_tests_ci() {
    log_info "Test suite in modalità CI/CD..."
    docker-compose run --rm richter-predictor python tests/run_tests.py --ci
    log_success "Test CI/CD completati!"
}

run_tests_quick() {
    log_info "Test rapidi (solo utilità)..."
    docker-compose run --rm richter-predictor python tests/run_tests.py --test utils --quiet
    log_success "Test rapidi completati!"
}

# Funzione per validazione completa pre-deploy
run_validation() {
    log_info "Validazione completa pre-deployment..."
    echo "🔍 Step 1: Test suite completa"
    docker-compose run --rm richter-predictor python tests/run_tests.py --ci
    
    echo "🔍 Step 2: Verifica preprocessing"
    docker-compose run --rm richter-predictor python -c "
from src.preprocessing.main_pipeline import RichterPreprocessingPipeline
import pandas as pd
import numpy as np

# Test rapido pipeline
pipeline = RichterPreprocessingPipeline()
pipeline.setup_preprocessors()
print(' Pipeline inizializzata correttamente')

# Test con dati dummy
dummy_data = pd.DataFrame({
    'geo_level_1_id': [1, 2, 3],
    'count_families': [1, 2, 3],
    'age': [10, 20, 30]
})
pipeline.fit(dummy_data)
result = pipeline.transform(dummy_data)
print(' Pipeline preprocessing OK')
print(f' Output shape validation: {type(result)}')
"
    
    echo " Step 3: Verifica TensorFlow"
    docker-compose run --rm richter-predictor python -c "
import tensorflow as tf
print(f' TensorFlow version: {tf.__version__}')
print(f' GPU available: {len(tf.config.list_physical_devices(\"GPU\"))} devices')
print(' TensorFlow working correctly')
"
    
    log_success "Validazione completa superata! "
}

# Funzione per training con test pre/post
run_training_with_validation() {
    log_info "Training con validazione pre/post..."
    
    echo " Pre-training validation"
    docker-compose run --rm richter-predictor python tests/run_tests.py --test preprocessing --quiet
    
    echo " Training modello"
    docker-compose run --rm richter-predictor python src/models/train_final_nested_cv.py

    echo " Post-training validation"
    docker-compose run --rm richter-predictor python tests/run_tests.py --test models --quiet
    
    log_success "Training con validazione completato!"
}

# Funzione per cleanup
cleanup() {
    log_info "Cleanup containers e immagini..."
    docker-compose down --volumes --remove-orphans
    docker system prune -f
    log_success "Cleanup completato!"
}

# Funzione per mostrare logs
show_logs() {
    log_info "Mostra logs dei container..."
    docker-compose logs -f
}

# Funzione per status
show_status() {
    log_info "Status dei container:"
    docker-compose ps
    echo ""
    log_info "Spazio utilizzato:"
    docker system df
}

# Menu principale - AGGIORNATO CON TEST
show_menu() {
    echo ""
    echo " COMANDI DISPONIBILI:"
    echo "1)  setup         - Setup iniziale (build + directory)"
    echo "2)  build         - Build Docker image"
    echo "3)  train         - Avvia training del modello"
    echo "4)  train-val     - Training con validazione pre/post"
    echo "5)  submit        - Crea submission finale"
    echo "6)  shell         - Shell interattiva nel container"
    echo "7)  analysis      - Esegui analisi dati"
    echo "8)  eda           - Genera EDA e visualizzazioni"
    echo "9)  test          - Test suite completa"
    echo "10) test-prep     - Test preprocessing only"
    echo "11) test-models   - Test modelli only"
    echo "12) test-utils    - Test utilità only"
    echo "13) test-ci       - Test suite CI/CD mode"
    echo "14) test-quick    - Test rapidi"
    echo "15) validate      - Validazione completa pre-deploy"
    echo "16) logs          - Mostra logs"
    echo "17) status        - Status container e spazio"
    echo "18) cleanup       - Cleanup completo"
    echo "19) help          - Mostra questo menu"
    echo "20) exit          - Esci"
    echo ""
}

# Parsing dei comandi - AGGIORNATO CON TEST
case "$1" in
    setup)
        check_docker
        setup_directories
        build_image
        log_success "Setup completato! Usa './docker-helper.sh help' per vedere i comandi."
        ;;
    build)
        check_docker
        build_image
        ;;
    train)
        check_docker
        run_training
        ;;
    train-val)
        check_docker
        run_training_with_validation
        ;;
    submit)
        check_docker
        run_submission
        ;;
    shell)
        check_docker
        run_shell
        ;;
    analysis)
        check_docker
        run_analysis
        ;;
    eda)
        check_docker
        run_eda
        ;;
    test)
        check_docker
        run_tests
        ;;
    test-prep)
        check_docker
        run_tests_preprocessing
        ;;
    test-models)
        check_docker
        run_tests_models
        ;;
    test-utils)
        check_docker
        run_tests_utils
        ;;
    test-ci)
        check_docker
        run_tests_ci
        ;;
    test-quick)
        check_docker
        run_tests_quick
        ;;
    validate)
        check_docker
        run_validation
        ;;
    logs)
        check_docker
        show_logs
        ;;
    status)
        check_docker
        show_status
        ;;
    cleanup)
        check_docker
        cleanup
        ;;
    help|--help|-h)
        show_menu
        ;;
    "")
        show_menu
        read -p "Seleziona comando (1-20): " choice
        case $choice in
            1) $0 setup ;;
            2) $0 build ;;
            3) $0 train ;;
            4) $0 train-val ;;
            5) $0 submit ;;
            6) $0 shell ;;
            7) $0 analysis ;;
            8) $0 eda ;;
            9) $0 test ;;
            10) $0 test-prep ;;
            11) $0 test-models ;;
            12) $0 test-utils ;;
            13) $0 test-ci ;;
            14) $0 test-quick ;;
            15) $0 validate ;;
            16) $0 logs ;;
            17) $0 status ;;
            18) $0 cleanup ;;
            19) $0 help ;;
            20) exit 0 ;;
            *) log_error "Opzione non valida!" ;;
        esac
        ;;
    *)
        log_error "Comando non riconosciuto: $1"
        show_menu
        exit 1
        ;;
esac