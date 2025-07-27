#!/bin/bash

# Docker Utility Script per Richter Predictor FIA
# Advanced Seismic Damage Classification System
# Version: 2.0.0 - Updated July 2025

set -e  # Exit su errori

# Colori per output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# 🏷️ Project info
PROJECT_NAME="Richter Predictor FIA"
PROJECT_VERSION="2.0.0"
DOCKER_IMAGE="richter-predictor:latest"
CONTAINER_NAME="richter-predictor-fia"

# 📊 Banner con informazioni sistema
show_banner() {
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║${WHITE}  🏗️  RICHTER PREDICTOR FIA - DOCKER UTILITY v2.0.0          ${BLUE}║${NC}"
    echo -e "${BLUE}║${CYAN}     Advanced Seismic Damage Classification System            ${BLUE}║${NC}"
    echo -e "${BLUE}║                                                              ║${NC}"
    echo -e "${BLUE}║${YELLOW}  📈 Current: F1-Score 0.736 | 🎯 Target: F1-Score 0.78+     ${BLUE}║${NC}"
    echo -e "${BLUE}║${PURPLE}  🧠 6 Neural Architectures | ⚙️ 280+ Engineered Features    ${BLUE}║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo
}

# 🛠️ Helper functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${CYAN}[STEP]${NC} $1"
}

# ✅ Verifica prerequisiti
check_prerequisites() {
    log_step "Verifica prerequisiti sistema..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker non trovato! Installa Docker prima di continuare."
        echo "📥 Download: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose non trovato! Installa Docker Compose."
        echo "📥 Download: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon non attivo! Avvia Docker prima di continuare."
        exit 1
    fi
    
    log_success "Prerequisiti verificati (Docker $(docker --version | cut -d' ' -f3 | cut -d',' -f1))"
}

# 📁 Crea directory necessarie
create_directories() {
    log_step "Creazione struttura directory..."
    
    local dirs=(
        "data/raw"
        "data/interim" 
        "models/single_models"
        "models/ensemble_models"
        "models/experiments"
        "reports/eda/figures"
        "reports/eda/tables"
        "reports/model_performance"
        "submissions"
        "logs"
        "tests/test_data"
    )
    
    for dir in "${dirs[@]}"; do
        if [[ ! -d "$dir" ]]; then
            mkdir -p "$dir"
            log_info "Creata directory: $dir"
        fi
    done
    
    log_success "Struttura directory pronta"
}

# 🔧 Setup permissions
setup_permissions() {
    log_step "Configurazione permessi..."
    
    # Make scripts executable
    chmod +x docker-helper.sh 2>/dev/null || true
    
    # Set appropriate permissions per directories
    find . -type d -name "logs" -exec chmod 755 {} \; 2>/dev/null || true
    find . -type d -name "models" -exec chmod 755 {} \; 2>/dev/null || true
    find . -type d -name "reports" -exec chmod 755 {} \; 2>/dev/null || true
    find . -type d -name "submissions" -exec chmod 755 {} \; 2>/dev/null || true
    
    log_success "Permessi configurati"
}

# 🐳 Build Docker image
build_image() {
    log_step "Build Docker image..."
    
    local build_args=""
    
    # GPU support detection
    if command -v nvidia-smi &> /dev/null; then
        log_info "🎮 GPU NVIDIA rilevata, enabling CUDA support"
        build_args="--build-arg CUDA_SUPPORT=true"
    fi
    
    # Build con cache optimization
    DOCKER_BUILDKIT=1 docker build $build_args \
        --tag $DOCKER_IMAGE \
        --progress=plain \
        . || {
        log_error "Build fallita!"
        exit 1
    }
    
    log_success "Docker image built: $DOCKER_IMAGE"
}

# 🚀 Setup completo
setup_complete() {
    show_banner
    check_prerequisites
    create_directories
    setup_permissions
    build_image
    
    echo
    log_success "🎉 Setup completo! Sistema pronto per l'uso."
    echo
    echo -e "${WHITE}📋 PROSSIMI PASSI:${NC}"
    echo -e "  🎯 Training:     ${CYAN}./docker-helper.sh train-nested${NC}"
    echo -e "  🧪 Testing:      ${CYAN}./docker-helper.sh test${NC}" 
    echo -e "  📊 EDA:          ${CYAN}./docker-helper.sh eda${NC}"
    echo -e "  📤 Submission:   ${CYAN}./docker-helper.sh submit${NC}"
    echo -e "  🖥️  Shell:        ${CYAN}./docker-helper.sh shell${NC}"
    echo -e "  📚 Help:         ${CYAN}./docker-helper.sh help${NC}"
    echo
}

# ═══════════════════════════════════════════════════════════════
# 🤖 TRAINING FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# 🎯 Training Nested CV Ensemble (Production)
train_nested_cv() {
    log_step "🎯 Training Ensemble con Nested CV (Produzione)..."
    log_info "Target: F1-Score ≥ 0.78 | Tempo stimato: 1-2 ore"
    log_info "Architetture: 6 networks | Features: 280+"
    
    # Verifica spazio disco
    check_disk_space 5  # 5GB minimum
    
    docker-compose --profile training run --rm richter-training-nested || {
        log_error "Training Nested CV fallito!"
        echo "💡 Suggerimenti:"
        echo "  - Verifica RAM disponibile (raccomandato: 12GB+)"
        echo "  - Prova training su subset: ./docker-helper.sh train-subset"
        echo "  - Check logs: ./docker-helper.sh logs"
        exit 1
    }
    
    log_success "🏆 Training Nested CV completato!"
    show_model_summary
}

# 🚀 Training Simple (Development)
train_simple() {
    log_step "🚀 Training modello singolo (Development)..."
    log_info "Modalità: Debug/Development | Tempo stimato: 10-15 minuti"
    
    docker-compose --profile training-simple run --rm richter-training-simple || {
        log_error "Training semplice fallito!"
        exit 1
    }
    
    log_success "✅ Training semplice completato!"
    show_model_summary
}

# ⚡ Training su subset (Testing)
train_subset() {
    log_step "⚡ Training rapido su subset..."
    log_info "Modalità: Testing | Tempo stimato: 2-3 minuti"
    
    docker-compose run --rm richter-predictor python test_nested_cv_subset.py || {
        log_error "Training subset fallito!"
        exit 1
    }
    
    log_success "⚡ Training subset completato!"
}

# ═══════════════════════════════════════════════════════════════
# 🧪 TESTING FUNCTIONS  
# ═══════════════════════════════════════════════════════════════

# 🔬 Test suite completa
run_tests_complete() {
    log_step "🔬 Esecuzione test suite completa..."
    log_info "Coverage: 95%+ | Tempo stimato: 10-15 minuti"
    
    docker-compose --profile testing run --rm richter-tests || {
        log_error "Test suite fallita!"
        echo "💡 Prova test specifici: ./docker-helper.sh test-core"
        exit 1
    }
    
    log_success "✅ Test suite completata!"
}

# ⚡ Test rapidi
run_tests_quick() {
    log_step "⚡ Test rapidi core components..."
    
    docker-compose run --rm richter-predictor python tests/run_tests.py --test core --quiet || {
        log_error "Test rapidi falliti!"
        exit 1
    }
    
    log_success "⚡ Test rapidi completati!"
}

# ⚙️ Test feature engineering
run_tests_feature_engineering() {
    log_step "⚙️ Test feature engineering modulare..."
    
    docker-compose run --rm richter-predictor python tests/test_modular_feature_engineering.py || {
        log_error "Test feature engineering falliti!"
        exit 1
    }
    
    log_success "✅ Test feature engineering completati!"
}

# 🤖 Test modelli ML
run_tests_models() {
    log_step "🤖 Test modelli e architetture..."
    
    docker-compose run --rm richter-predictor python tests/test_models.py || {
        log_error "Test modelli falliti!"
        exit 1
    }
    
    log_success "✅ Test modelli completati!"
}

# 🔒 Test anti-leakage
run_tests_antileakage() {
    log_step "🔒 Test sistema anti-leakage..."
    
    docker-compose run --rm richter-predictor python tests/test_nested_cv_trainer.py || {
        log_error "Test anti-leakage falliti!"
        exit 1
    }
    
    log_success "🔒 Test anti-leakage completati!"
}

# ✅ Validazione completa pre-deploy
run_validation() {
    log_step "✅ Validazione completa sistema..."
    log_info "Include: Tests + Model check + Data validation"
    
    echo
    echo "🔬 1/4: Test suite core..."
    run_tests_quick
    
    echo
    echo "⚙️ 2/4: Validazione feature engineering..."
    docker-compose run --rm richter-predictor python -c "
from feature_engineering import AdvancedFeatureEngineer
import pandas as pd
print('🧠 Testing modular feature engineering...')
engineer = AdvancedFeatureEngineer({'verbose': False})
print('✅ AdvancedFeatureEngineer import OK')
"
    
    echo
    echo "📊 3/4: Verifica dati disponibili..."
    docker-compose run --rm richter-predictor python -c "
import pandas as pd
from pathlib import Path
data_dir = Path('data/raw')
files = ['train_values.csv', 'train_labels.csv']
for f in files:
    if (data_dir / f).exists():
        df = pd.read_csv(data_dir / f)
        print(f'✅ {f}: {df.shape}')
    else:
        print(f'❌ {f}: Missing')
"
        
    echo
    echo "🤖 4/4: Test architetture modelli..."
    docker-compose run --rm richter-predictor python -c "
from models.ensemble_architectures import EnsembleArchitectures
ensemble = EnsembleArchitectures(100, 3)
archs = ensemble.get_available_architectures()
print(f'✅ Architetture disponibili: {len(archs)}')
print(f'   {archs}')
"
    
    echo
    log_success "🎉 Validazione completata! Sistema pronto per produzione."
}

# ═══════════════════════════════════════════════════════════════
# 📊 ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# 📈 EDA completa
run_eda() {
    log_step "📈 Esecuzione EDA completa..."
    log_info "Genera: Visualizzazioni + Report + Statistiche"
    
    docker-compose --profile eda run --rm richter-eda || {
        log_error "EDA fallita!"
        exit 1
    }
    
    log_success "📊 EDA completata! Controlla reports/eda/"
    
    # Show generated files
    if [[ -d "reports/eda" ]]; then
        echo
        echo "📁 File generati:"
        find reports/eda -type f -name "*.png" -o -name "*.csv" -o -name "*.json" | head -10
    fi
}

# 🔍 Debug sparsità
debug_sparsity() {
    log_step "🔍 Debug sparsità dati..."
    
    docker-compose run --rm richter-predictor python debug_sparsity.py || {
        log_error "Debug sparsity fallito!"
        exit 1
    }
    
    log_success "🔍 Debug sparsity completato!"
}

# 📊 Analisi automatica features
run_analysis() {
    log_step "📊 Analisi automatica features..."
    
    docker-compose run --rm richter-predictor python src/data/data_analysis.py || {
        log_error "Analisi features fallita!"
        exit 1
    }
    
    log_success "📊 Analisi features completata!"
}

# ═══════════════════════════════════════════════════════════════
# 📤 SUBMISSION FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# 📤 Genera submission
generate_submission() {
    log_step "📤 Generazione submission DrivenData..."
    
    # Check modelli disponibili
    if [[ ! -d "models" ]] || [[ -z "$(ls -A models/ 2>/dev/null)" ]]; then
        log_warning "Nessun modello trovato in models/"
        echo "💡 Esegui prima training: ./docker-helper.sh train-simple"
        exit 1
    fi
    
    docker-compose --profile submission run --rm richter-submission || {
        log_error "Generazione submission fallita!"
        exit 1
    }
    
    log_success "📤 Submission generata!"
    
    # Show submission files
    if [[ -d "submissions" ]]; then
        echo
        echo "📁 File submission:"
        ls -la submissions/*.csv 2>/dev/null | tail -5 || echo "Nessun file .csv trovato"
    fi
}

# ═══════════════════════════════════════════════════════════════
# 🛠️ UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

# 🖥️ Shell interattiva
run_shell() {
    log_step "🖥️  Avvio shell interattiva..."
    
    docker-compose run --rm richter-predictor bash || {
        log_error "Shell fallita!"
        exit 1
    }
}

# 📝 Visualizza logs
show_logs() {
    log_step "📝 Visualizzazione logs..."
    
    # System logs
    docker-compose logs --tail=50 richter-predictor 2>/dev/null || true
    
    # Application logs
    if [[ -d "logs" ]]; then
        echo
        echo "📁 Application logs:"
        find logs -name "*.log" -type f -exec tail -10 {} \; 2>/dev/null || echo "Nessun log applicativo trovato"
    fi
}

# 📊 Status sistema
show_status() {
    log_step "📊 Status sistema Docker..."
    
    echo
    echo "🐳 Container attivi:"
    docker ps --filter "name=richter" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo
    echo "🖼️  Images disponibili:"
    docker images --filter "reference=richter*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    echo
    echo "💾 Utilizzo spazio Docker:"
    docker system df
    
    # Project status
    echo
    echo "📁 Status progetto:"
    [[ -d "data/raw" ]] && echo "  ✅ Data directory" || echo "  ❌ Data directory"
    [[ -d "models" ]] && echo "  ✅ Models directory" || echo "  ❌ Models directory"
    [[ -f "requirements.txt" ]] && echo "  ✅ Requirements file" || echo "  ❌ Requirements file"
    
    # Model summary
    show_model_summary
}

# 🏆 Mostra summary modelli
show_model_summary() {
    if [[ -d "models" ]] && [[ -n "$(ls -A models/ 2>/dev/null)" ]]; then
        echo
        echo "🏆 Modelli disponibili:"
        find models -name "*.keras" -o -name "*.h5" -o -name "*.pkl" | head -5 | while read -r model; do
            size=$(du -h "$model" 2>/dev/null | cut -f1)
            echo "  📊 $(basename "$model") ($size)"
        done
    fi
}

# 💾 Check spazio disco
check_disk_space() {
    local required_gb=${1:-5}
    local available_gb=$(df . | awk 'NR==2 {print int($4/1024/1024)}')
    
    if [[ $available_gb -lt $required_gb ]]; then
        log_warning "Spazio disco insufficiente: ${available_gb}GB < ${required_gb}GB richiesti"
        echo "💡 Liberare spazio o usare: ./docker-helper.sh cleanup"
        return 1
    fi
    
    log_info "Spazio disco OK: ${available_gb}GB disponibili"
}

# 🧹 Cleanup completo
cleanup_system() {
    log_step "🧹 Cleanup sistema Docker..."
    
    # Stop containers
    docker-compose down 2>/dev/null || true
    
    # Remove containers
    docker ps -a --filter "name=richter" -q | xargs -r docker rm 2>/dev/null || true
    
    # Remove images (with confirmation)
    read -p "🗑️  Rimuovere images Docker? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker images --filter "reference=richter*" -q | xargs -r docker rmi 2>/dev/null || true
        log_success "Images rimosse"
    fi
    
    # System cleanup
    docker system prune -f 2>/dev/null || true
    
    log_success "🧹 Cleanup completato!"
}
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

# Funzione per training ensemble modulare (UPDATED)
run_training() {
    log_info "Avvio training ensemble con feature engineering modulare..."
    log_info "Questo utilizza la nuova pipeline con 6 moduli di feature engineering"
    docker-compose run --rm richter-predictor python src/models/train_nested_cv_ensemble.py
    log_success "Training ensemble modulare completato!"
}

# Funzione per submission con nuovo sistema modulare
run_submission() {
    log_info "Generazione submission con modello migliore disponibile..."
    docker-compose run --rm richter-predictor python src/create_submission.py --model_type=single
    log_success "Submission generata! Check submissions/ directory"
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
# ═══════════════════════════════════════════════════════════════
# 📚 HELP FUNCTION
# ═══════════════════════════════════════════════════════════════

show_help() {
    show_banner
    
    echo -e "${WHITE}📋 COMANDI DISPONIBILI:${NC}"
    echo
    
    echo -e "${CYAN}🔧 SETUP E CONFIGURAZIONE:${NC}"
    echo -e "  ${WHITE}setup${NC}           Setup completo (raccomandato per primo uso)"
    echo -e "  ${WHITE}build${NC}           Build Docker image"
    echo -e "  ${WHITE}clean${NC}           Cleanup completo sistema"
    echo
    
    echo -e "${CYAN}🤖 TRAINING MODELLI:${NC}"
    echo -e "  ${WHITE}train-nested${NC}    🎯 Training ensemble Nested CV (produzione, 1-2h)"
    echo -e "  ${WHITE}train-simple${NC}    🚀 Training modello singolo (debug, 15min)"
    echo -e "  ${WHITE}train-subset${NC}    ⚡ Training rapido su subset (testing, 3min)"
    echo
    
    echo -e "${CYAN}🧪 TESTING E VALIDAZIONE:${NC}"
    echo -e "  ${WHITE}test${NC}            🔬 Test suite completa (95% coverage, 15min)"
    echo -e "  ${WHITE}test-quick${NC}      ⚡ Test rapidi core components (3min)"
    echo -e "  ${WHITE}test-fe${NC}         ⚙️ Test feature engineering modulare"
    echo -e "  ${WHITE}test-models${NC}     🤖 Test modelli e architetture"
    echo -e "  ${WHITE}test-antileakage${NC} 🔒 Test sistema anti-leakage"
    echo -e "  ${WHITE}validate${NC}        ✅ Validazione completa pre-deploy"
    echo
    
    echo -e "${CYAN}📊 ANALISI E EDA:${NC}"
    echo -e "  ${WHITE}eda${NC}             📈 EDA completa con visualizzazioni"
    echo -e "  ${WHITE}analysis${NC}        📊 Analisi automatica features"
    echo -e "  ${WHITE}debug-sparse${NC}    🔍 Debug sparsità dati"
    echo
    
    echo -e "${CYAN}📤 SUBMISSION E DEPLOY:${NC}"
    echo -e "  ${WHITE}submit${NC}          📤 Genera submission DrivenData"
    echo
    
    echo -e "${CYAN}🛠️ UTILITIES:${NC}"
    echo -e "  ${WHITE}shell${NC}           🖥️ Shell interattiva container"
    echo -e "  ${WHITE}logs${NC}            📝 Visualizza logs sistema"
    echo -e "  ${WHITE}status${NC}          📊 Status container e modelli"
    echo -e "  ${WHITE}help${NC}            📚 Mostra questo help"
    echo
    
    echo -e "${WHITE}💡 ESEMPI D'USO:${NC}"
    echo -e "  ${GRAY}# Setup iniziale completo${NC}"
    echo -e "  ${CYAN}./docker-helper.sh setup${NC}"
    echo
    echo -e "  ${GRAY}# Workflow sviluppo rapido${NC}"
    echo -e "  ${CYAN}./docker-helper.sh train-simple && ./docker-helper.sh test-quick${NC}"
    echo
    echo -e "  ${GRAY}# Training produzione completo${NC}"
    echo -e "  ${CYAN}./docker-helper.sh train-nested && ./docker-helper.sh validate${NC}"
    echo
    echo -e "  ${GRAY}# Analisi dati completa${NC}"
    echo -e "  ${CYAN}./docker-helper.sh eda && ./docker-helper.sh analysis${NC}"
    echo
    
    echo -e "${WHITE}WORKFLOW RACCOMANDATO:${NC}"
    echo -e "  1  ${CYAN}./docker-helper.sh setup${NC}         # Setup iniziale"
    echo -e "  2  ${CYAN}./docker-helper.sh eda${NC}           # Esplora i dati"
    echo -e "  3  ${CYAN}./docker-helper.sh train-simple${NC}  # Test training"
    echo -e "  4  ${CYAN}./docker-helper.sh test${NC}          # Valida sistema"
    echo -e "  5  ${CYAN}./docker-helper.sh train-nested${NC}  # Training produzione"
    echo -e "  6  ${CYAN}./docker-helper.sh submit${NC}        # Genera submission"
    echo
    
    echo -e "${WHITE}SUPPORTO:${NC}"
    echo -e "  Issues: https://github.com/DanieleLimongi/richter-predictor-fia/issues"
    echo -e "  Docs:   README.md del progetto"
    echo -e "  Help:   ./docker-helper.sh help"
    echo
}

# ═══════════════════════════════════════════════════════════════
# 🎛️ MAIN COMMAND DISPATCHER
# ═══════════════════════════════════════════════════════════════

main() {
    local command=${1:-help}
    
    # Show banner for all commands except help
    if [[ "$command" != "help" ]]; then
        show_banner
    fi
    
    case $command in
        # 🔧 Setup e configurazione
        "setup")
            setup_complete
            ;;
        "build")
            check_prerequisites
            build_image
            ;;
        "clean"|"cleanup")
            cleanup_system
            ;;
            
        # 🤖 Training
        "train-nested"|"training-nested")
            train_nested_cv
            ;;
        "train-simple"|"training-simple")
            train_simple
            ;;
        "train-subset"|"training-subset")
            train_subset
            ;;
            
        # 🧪 Testing
        "test"|"tests")
            run_tests_complete
            ;;
        "test-quick"|"tests-quick")
            run_tests_quick
            ;;
        "test-fe"|"test-feature-engineering")
            run_tests_feature_engineering
            ;;
        "test-models")
            run_tests_models
            ;;
        "test-antileakage"|"test-anti-leakage")
            run_tests_antileakage
            ;;
        "validate"|"validation")
            run_validation
            ;;
            
        # 📊 Analisi
        "eda")
            run_eda
            ;;
        "analysis"|"analyze")
            run_analysis
            ;;
        "debug-sparse"|"debug-sparsity")
            debug_sparsity
            ;;
            
        # 📤 Submission
        "submit"|"submission")
            generate_submission
            ;;
            
        # 🛠️ Utilities
        "shell"|"bash")
            run_shell
            ;;
        "logs"|"log")
            show_logs
            ;;
        "status"|"info")
            show_status
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
            
        # ❓ Unknown command
        *)
            log_error "Comando sconosciuto: '$command'"
            echo
            echo -e "${WHITE}💡 Comandi disponibili:${NC}"
            echo -e "  ./docker-helper.sh help    # Aiuto completo"
            echo -e "  ./docker-helper.sh setup   # Setup iniziale"
            echo -e "  ./docker-helper.sh status  # Status sistema"
            echo
            exit 1
            ;;
    esac
}

# ═══════════════════════════════════════════════════════════════
# 🚀 SCRIPT EXECUTION
# ═══════════════════════════════════════════════════════════════

# Trap per cleanup in caso di interruzione
trap 'echo -e "\n${YELLOW}⚠️ Script interrotto dall'\''utente${NC}"; exit 130' INT

# Esegui comando principale
main "$@"
print('✅ AdvancedFeatureEngineer initialized')

# Test with realistic dummy data
dummy_data = pd.DataFrame({
    'geo_level_1_id': [1, 2, 3, 4, 5],
    'geo_level_2_id': [10, 20, 30, 40, 50],
    'geo_level_3_id': [100, 200, 300, 400, 500],
    'count_families': [1, 2, 3, 2, 1],
    'age': [10, 20, 30, 15, 25],
    'area_percentage': [5, 10, 8, 12, 6],
    'height_percentage': [3, 5, 4, 6, 4],
    'count_floors_pre_eq': [1, 2, 3, 2, 1],
    'land_surface_condition': ['t', 'o', 't', 'n', 't'],
    'foundation_type': ['r', 'h', 'r', 'w', 'i'],
    'roof_type': ['n', 'q', 'x', 'n', 'q'],
    'damage_grade': [1, 2, 3, 2, 1]
})

print(f'Input shape: {dummy_data.shape}')
result = engineer.fit_transform(dummy_data, 'damage_grade')
print(f'Output shape: {result.shape}')
print(f'✅ Feature engineering: {dummy_data.shape[1]} → {result.shape[1]} features')
print(f'✅ Features added: +{result.shape[1] - dummy_data.shape[1]}')

# Test transform (simulate test data)
test_result = engineer.transform(dummy_data.drop('damage_grade', axis=1))
print(f'✅ Transform test: {test_result.shape}')
print('🎯 Modular feature engineering validation PASSED')
"
    
    echo "Step 3: Verifica TensorFlow e GPU"
    docker-compose run --rm richter-predictor python -c "
import tensorflow as tf
print(f'🔧 TensorFlow version: {tf.__version__}')
gpu_devices = tf.config.list_physical_devices('GPU')
print(f'🖥️  GPU devices: {len(gpu_devices)}')
if gpu_devices:
    for i, gpu in enumerate(gpu_devices):
        print(f'   GPU {i}: {gpu}')
else:
    print('   CPU-only mode (GPU optional)')
print('✅ TensorFlow working correctly')
"
    
    echo "Step 4: Verifica architetture ensemble"
    docker-compose run --rm richter-predictor python -c "
from models.ensemble_architectures import EnsembleArchitectures
print('🤖 Testing ensemble architectures...')
ensemble = EnsembleArchitectures(100, 3)
architectures = ensemble.get_available_architectures()
print(f'✅ Available architectures: {len(architectures)}')
for arch in architectures:
    print(f'   - {arch}')
print('🎯 Ensemble architectures validation PASSED')
"
    
    log_success "🎉 Validazione completa sistema modulare SUPERATA!"
}

# Funzione per training con nested CV
run_training_nested_cv() {
    log_info "Training con Nested CV (anti-leakage)..."
    docker-compose run --rm richter-predictor python src/models/train_nested_cv_ensemble.py
    log_success "Training Nested CV completato!"
}

# Funzione per training semplice
run_training_simple() {
    log_info "Training semplice holdout..."
    docker-compose run --rm richter-predictor python src/models/train_simple_holdout.py
    log_success "Training semplice completato!"
}

# Funzione per test subset nested CV
run_test_nested_cv() {
    log_info "Test Nested CV su subset..."
    docker-compose run --rm richter-predictor python test_nested_cv_subset.py
    log_success "Test Nested CV completato!"
}

# Funzione per debug sparsity
run_debug_sparsity() {
    log_info "Debug data sparsity..."
    docker-compose run --rm richter-predictor python debug_sparsity.py
    log_success "Debug sparsity completato!"
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

# Menu principale - AGGIORNATO PER SISTEMA MODULARE
show_menu() {
    echo ""
    echo "🚀 COMANDI DISPONIBILI - RICHTER PREDICTOR FIA:"
    echo "==============================================="
    echo ""
    echo "📦 SETUP & BUILD:"
    echo "1)  setup         - Setup iniziale completo (build + directory)"
    echo "2)  build         - Build Docker image"
    echo ""
    echo "🤖 TRAINING MODELLI:"
    echo "3)  train         - Training ensemble modulare (6 architetture + 6 moduli FE)"
    echo "4)  train-nested  - Training con Nested CV anti-leakage"
    echo "5)  train-simple  - Training semplice holdout (debug)"
    echo "6)  test-nested   - Test Nested CV su subset"
    echo ""
    echo "📊 ANALISI & DEBUG:"
    echo "7)  analysis      - Analisi automatica dati"
    echo "8)  eda           - EDA completa con visualizzazioni"
    echo "9)  debug-sparse  - Debug data sparsity"
    echo ""
    echo "🧪 TESTING:"
    echo "10) test          - Test suite completa"
    echo "11) test-prep     - Test feature engineering modulare"
    echo "12) test-models   - Test architetture ensemble"
    echo "13) test-utils    - Test utilities"
    echo "14) test-ci       - Test suite CI/CD mode"
    echo "15) test-quick    - Test rapidi (2-3 min)"
    echo "16) validate      - Validazione completa sistema modulare"
    echo ""
    echo "📤 SUBMISSION & DEPLOY:"
    echo "17) submit        - Genera submission con modello migliore"
    echo "18) shell         - Shell interattiva nel container"
    echo ""
    echo "🔧 UTILITIES:"
    echo "19) logs          - Mostra logs container"
    echo "20) status        - Status container e spazio"
    echo "21) cleanup       - Cleanup completo"
    echo "22) help          - Mostra questo menu"
    echo "23) exit          - Esci"
    echo ""
    echo "💡 Quick Start: ./docker-helper.sh setup && ./docker-helper.sh train-nested"
    echo ""
}

# Parsing dei comandi - AGGIORNATO CON TEST
case "$1" in
    setup)
        check_docker
        setup_directories
        build_image
        log_success "🎉 Setup completato! Sistema modulare pronto per training ensemble."
        echo ""
        echo "🚀 Next Steps:"
        echo "   ./docker-helper.sh validate     # Verifica sistema completo"
        echo "   ./docker-helper.sh train-nested # Training ensemble modulare"
        echo "   ./docker-helper.sh submit       # Genera submission"
        ;;
    build)
        check_docker
        build_image
        ;;
    train)
        check_docker
        run_training
        ;;
    train-nested)
        check_docker
        run_training_nested_cv
        ;;
    train-simple)
        check_docker
        run_training_simple
        ;;
    test-nested)
        check_docker
        run_test_nested_cv
        ;;
    debug-sparse)
        check_docker
        run_debug_sparsity
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
        read -p "Seleziona comando (1-23): " choice
        case $choice in
            1) $0 setup ;;
            2) $0 build ;;
            3) $0 train ;;
            4) $0 train-nested ;;
            5) $0 train-simple ;;
            6) $0 test-nested ;;
            7) $0 debug-sparse ;;
            8) $0 submit ;;
            9) $0 shell ;;
            10) $0 analysis ;;
            11) $0 eda ;;
            12) $0 test ;;
            13) $0 test-prep ;;
            14) $0 test-models ;;
            15) $0 test-utils ;;
            16) $0 test-ci ;;
            17) $0 test-quick ;;
            18) $0 validate ;;
            19) $0 logs ;;
            20) $0 status ;;
            21) $0 cleanup ;;
            22) $0 help ;;
            23) exit 0 ;;
            *) log_error "Opzione non valida!" ;;
        esac
        ;;
    *)
        log_error "Comando non riconosciuto: $1"
        show_menu
        exit 1
        ;;
esac