#!/bin/bash

# Richter Predictor FIA - Docker Helper Script
# Version: 3.0.0 - Completely rewritten and cleaned up

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
GRAY='\033[0;37m'
NC='\033[0m'

# Project configuration
PROJECT_NAME="Richter Predictor FIA"
PROJECT_VERSION="3.0.0"
DOCKER_IMAGE="richter-predictor:latest"
CONTAINER_NAME="richter-predictor-fia"

# Logging functions
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

# Show banner
show_banner() {
    echo -e "${BLUE}=====================================================${NC}"
    echo -e "${WHITE}  $PROJECT_NAME - Docker Helper v$PROJECT_VERSION${NC}"
    echo -e "${GRAY}  Advanced Seismic Damage Classification System${NC}"
    echo -e "${BLUE}=====================================================${NC}"
    echo
}

# Check prerequisites
check_prerequisites() {
    log_step "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker not found! Install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose not found! Install Docker Compose."
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon not running! Start Docker first."
        exit 1
    fi
    
    log_success "Prerequisites checked"
}

# Create directories
create_directories() {
    log_step "Creating directory structure..."
    
    local dirs=(
        "data/raw" "data/interim"
        "models" "reports" "submissions" "logs"
        "tests/test_data"
    )
    
    for dir in "${dirs[@]}"; do
        mkdir -p "$dir"
    done
    
    log_success "Directory structure created"
}

# Setup permissions
setup_permissions() {
    log_step "Setting up permissions..."
    chmod +x docker-helper.sh 2>/dev/null || true
    log_success "Permissions configured"
}

# Build Docker image
build_image() {
    log_step "Building Docker image..."
    
    docker-compose build richter-predictor || {
        log_error "Build failed!"
        exit 1
    }
    
    log_success "Docker image built: $DOCKER_IMAGE"
}

# Complete setup
setup_complete() {
    show_banner
    check_prerequisites
    create_directories
    setup_permissions
    build_image
    
    echo
    log_success "Setup complete! System ready."
    echo
    echo -e "${WHITE}Next steps:${NC}"
    echo -e "  ${CYAN}./docker-helper.sh train-simple${NC}  # Quick training test"
    echo -e "  ${CYAN}./docker-helper.sh test core${NC}     # Run core tests"
    echo -e "  ${CYAN}./docker-helper.sh eda${NC}           # Explore data"
    echo -e "  ${CYAN}./docker-helper.sh help${NC}          # Show all commands"
    echo
}

# Training functions
train_nested() {
    log_step "Starting Nested CV training..."
    log_info "Target: F1-Score >= 0.78 | Estimated time: 1-2 hours"
    
    docker-compose --profile training run --rm richter-training-nested || {
        log_error "Nested CV training failed!"
        exit 1
    }
    
    log_success "Nested CV training completed!"
    show_model_summary
}

train_simple() {
    log_step "Starting simple training..."
    log_info "Mode: Development | Estimated time: 10-15 minutes"
    
    docker-compose --profile training-simple run --rm richter-training-simple || {
        log_error "Simple training failed!"
        exit 1
    }
    
    log_success "Simple training completed!"
    show_model_summary
}

# Test functions with submenu
run_tests() {
    local test_type=${1:-"menu"}
    
    case $test_type in
        "menu")
            show_test_menu
            ;;
        "all")
            log_step "Running complete test suite..."
            docker-compose --profile testing run --rm richter-tests || {
                log_error "Test suite failed!"
                exit 1
            }
            log_success "Complete test suite passed!"
            ;;
        "core")
            log_step "Running core component tests..."
            docker-compose run --rm richter-predictor python tests/run_tests.py --test core || {
                log_error "Core tests failed!"
                exit 1
            }
            log_success "Core tests passed!"
            ;;
        "features")
            log_step "Running feature engineering tests..."
            docker-compose run --rm richter-predictor python tests/test_modular_feature_engineering.py || {
                log_error "Feature engineering tests failed!"
                exit 1
            }
            log_success "Feature engineering tests passed!"
            ;;
        "models")
            log_step "Running model tests..."
            docker-compose run --rm richter-predictor python tests/test_models.py || {
                log_error "Model tests failed!"
                exit 1
            }
            log_success "Model tests passed!"
            ;;
        "ensemble")
            log_step "Running ensemble tests..."
            docker-compose run --rm richter-predictor python tests/test_ensemble.py || {
                log_error "Ensemble tests failed!"
                exit 1
            }
            log_success "Ensemble tests passed!"
            ;;
        "integration")
            log_step "Running integration tests..."
            docker-compose run --rm richter-predictor python tests/test_integration.py || {
                log_error "Integration tests failed!"
                exit 1
            }
            log_success "Integration tests passed!"
            ;;
        "quick")
            log_step "Running quick tests..."
            docker-compose run --rm richter-predictor python tests/run_tests.py --test core --quick || {
                log_error "Quick tests failed!"
                exit 1
            }
            log_success "Quick tests passed!"
            ;;
        *)
            log_error "Unknown test type: $test_type"
            show_test_menu
            ;;
    esac
}

# Test submenu
show_test_menu() {
    echo
    echo -e "${WHITE}TEST MENU - Available Test Suites:${NC}"
    echo
    echo -e "${CYAN}Core Tests:${NC}"
    echo -e "  ${WHITE}core${NC}         Core functionality tests (fast)"
    echo -e "  ${WHITE}quick${NC}        Quick smoke tests (fastest)"
    echo
    echo -e "${CYAN}Component Tests:${NC}"
    echo -e "  ${WHITE}features${NC}     Feature engineering tests"
    echo -e "  ${WHITE}models${NC}       Model architecture tests"
    echo -e "  ${WHITE}ensemble${NC}     Ensemble system tests"
    echo
    echo -e "${CYAN}Complete Tests:${NC}"
    echo -e "  ${WHITE}integration${NC}  End-to-end integration tests"
    echo -e "  ${WHITE}all${NC}          Complete test suite (comprehensive)"
    echo
    echo -e "${WHITE}Usage:${NC}"
    echo -e "  ${CYAN}./docker-helper.sh test <type>${NC}"
    echo -e "  ${CYAN}./docker-helper.sh test${NC}        # Show this menu"
    echo
    echo -e "${WHITE}Examples:${NC}"
    echo -e "  ${GRAY}./docker-helper.sh test core        # Run core tests${NC}"
    echo -e "  ${GRAY}./docker-helper.sh test all         # Run all tests${NC}"
    echo
}

# Analysis functions
run_eda() {
    log_step "Running EDA analysis..."
    
    docker-compose --profile eda run --rm richter-eda || {
        log_error "EDA failed!"
        exit 1
    }
    
    log_success "EDA completed! Check reports/eda/"
    
    if [[ -d "reports/eda" ]]; then
        echo
        echo "Generated files:"
        find reports/eda -type f -name "*.png" -o -name "*.csv" | head -5
    fi
}

run_analysis() {
    log_step "Running data analysis..."
    
    docker-compose run --rm richter-predictor python src/data/data_analysis.py || {
        log_error "Data analysis failed!"
        exit 1
    }
    
    log_success "Data analysis completed!"
}

debug_sparsity() {
    log_step "Running sparsity analysis..."
    
    docker-compose run --rm richter-predictor python src/data/sparsity_analysis.py || {
        log_error "Sparsity analysis failed!"
        exit 1
    }
    
    log_success "Sparsity analysis completed! Check reports/sparsity/"
}

# Submission function
generate_submission() {
    local submission_type=${1:-"menu"}
    
    # Show submission menu if no type specified
    if [[ "$submission_type" == "menu" ]]; then
        echo
        echo -e "${CYAN}SUBMISSION GENERATOR${NC}"
        echo -e "${WHITE}Select submission type:${NC}"
        echo
        echo -e "  ${WHITE}1${NC} - Simple model submission (single best model)"
        echo -e "  ${WHITE}2${NC} - Ensemble submission (nested CV ensemble)"
        echo -e "  ${WHITE}3${NC} - List available models"
        echo
        read -p "Enter choice [1-3]: " choice
        
        case $choice in
            1) submission_type="simple" ;;
            2) submission_type="ensemble" ;;
            3) submission_type="list" ;;
            *) 
                log_error "Invalid choice"
                exit 1
                ;;
        esac
    fi
    
    # Check for models
    if [[ ! -d "models" ]]; then
        log_warning "No models directory found"
        log_info "Run training first: ./docker-helper.sh train-simple or train-nested"
        exit 1
    fi
    
    case $submission_type in
        "simple")
            log_step "Generating simple model submission..."
            
            if [[ ! -d "models/simple_models" ]] || [[ -z "$(ls -A models/simple_models/ 2>/dev/null)" ]]; then
                log_warning "No simple models found in models/simple_models/"
                log_info "Run simple training first: ./docker-helper.sh train-simple"
                exit 1
            fi
            
            docker-compose run --rm richter-app python src/create_submission_simple.py || {
                log_error "Simple submission generation failed!"
                exit 1
            }
            ;;
            
        "ensemble")
            log_step "Generating ensemble submission..."
            
            if [[ ! -d "models/nested_models" ]] || [[ -z "$(ls -A models/nested_models/ 2>/dev/null)" ]]; then
                log_warning "No ensemble models found in models/nested_models/"
                log_info "Run nested training first: ./docker-helper.sh train-nested"
                exit 1
            fi
            
            docker-compose run --rm richter-app python src/create_submission_ensemble.py || {
                log_error "Ensemble submission generation failed!"
                exit 1
            }
            ;;
            
        "list")
            log_step "Listing available models..."
            echo
            echo -e "${CYAN}Simple Models:${NC}"
            docker-compose run --rm richter-app python src/create_submission_simple.py --list-models
            echo
            echo -e "${CYAN}Ensemble Models:${NC}"
            docker-compose run --rm richter-app python src/create_submission_ensemble.py --list-ensembles
            return 0
            ;;
            
        *)
            log_error "Unknown submission type: $submission_type"
            exit 1
            ;;
    esac
    
    log_success "Submission generated!"
    
    if [[ -d "submissions" ]]; then
        echo
        echo "Latest submission files:"
        ls -la submissions/*.csv 2>/dev/null | tail -5 || echo "No CSV files found"
    fi
}

# Utility functions
run_shell() {
    log_step "Starting interactive shell..."
    docker-compose run --rm richter-predictor bash
}

show_logs() {
    log_step "Showing logs..."
    docker-compose logs --tail=50 richter-predictor 2>/dev/null || true
    
    if [[ -d "logs" ]]; then
        echo
        echo "Application logs:"
        find logs -name "*.log" -type f -exec tail -5 {} \; 2>/dev/null || echo "No application logs found"
    fi
}

show_status() {
    log_step "System status..."
    
    echo
    echo "Docker containers:"
    docker ps --filter "name=richter" --format "table {{.Names}}\t{{.Status}}"
    
    echo
    echo "Docker images:"
    docker images --filter "reference=richter*" --format "table {{.Repository}}\t{{.Tag}}\t{{.Size}}"
    
    echo
    echo "Disk usage:"
    docker system df
    
    show_model_summary
}

show_model_summary() {
    if [[ -d "models" ]] && [[ -n "$(ls -A models/ 2>/dev/null)" ]]; then
        echo
        echo "Available models:"
        find models -name "*.keras" -o -name "*.h5" | head -5 | while read -r model; do
            size=$(du -h "$model" 2>/dev/null | cut -f1)
            echo "  $(basename "$model") ($size)"
        done
    fi
}

cleanup_system() {
    log_step "Cleaning up Docker system..."
    
    docker-compose down 2>/dev/null || true
    docker ps -a --filter "name=richter" -q | xargs -r docker rm 2>/dev/null || true
    
    read -p "Remove Docker images? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker images --filter "reference=richter*" -q | xargs -r docker rmi 2>/dev/null || true
        log_success "Images removed"
    fi
    
    docker system prune -f 2>/dev/null || true
    log_success "Cleanup completed!"
}

# Help function
show_help() {
    show_banner
    
    echo -e "${WHITE}AVAILABLE COMMANDS:${NC}"
    echo
    
    echo -e "${CYAN}Setup & Build:${NC}"
    echo -e "  ${WHITE}setup${NC}           Complete system setup"
    echo -e "  ${WHITE}build${NC}           Build Docker image"
    echo -e "  ${WHITE}clean${NC}           Clean up system"
    echo
    
    echo -e "${CYAN}Training:${NC}"
    echo -e "  ${WHITE}train-nested${NC}    Nested CV ensemble training (1-2h)"
    echo -e "  ${WHITE}train-simple${NC}    Simple model training (15min)"
    echo
    
    echo -e "${CYAN}Testing:${NC}"
    echo -e "  ${WHITE}test [type]${NC}     Run tests (see test menu for types)"
    echo -e "  ${WHITE}test${NC}            Show test menu"
    echo
    
    echo -e "${CYAN}Analysis:${NC}"
    echo -e "  ${WHITE}eda${NC}             Exploratory data analysis"
    echo -e "  ${WHITE}analysis${NC}        Data analysis"
    echo -e "  ${WHITE}debug-sparse${NC}    Data sparsity analysis"
    echo
    
    echo -e "${CYAN}Utilities:${NC}"
    echo -e "  ${WHITE}submit [type]${NC}   Generate submission (simple/ensemble/list)"
    echo -e "  ${WHITE}submit${NC}          Show submission menu"
    echo -e "  ${WHITE}shell${NC}           Interactive shell"
    echo -e "  ${WHITE}logs${NC}            Show logs"
    echo -e "  ${WHITE}status${NC}          System status"
    echo -e "  ${WHITE}help${NC}            Show this help"
    echo
    
    echo -e "${WHITE}Examples:${NC}"
    echo -e "  ${GRAY}./docker-helper.sh setup               # Initial setup${NC}"
    echo -e "  ${GRAY}./docker-helper.sh train-simple        # Quick training${NC}"
    echo -e "  ${GRAY}./docker-helper.sh test core           # Run core tests${NC}"
    echo -e "  ${GRAY}./docker-helper.sh eda                 # Data analysis${NC}"
    echo
    
    echo -e "${WHITE}Workflow:${NC}"
    echo -e "  1. ${CYAN}./docker-helper.sh setup${NC}         # Setup system"
    echo -e "  2. ${CYAN}./docker-helper.sh eda${NC}           # Explore data"
    echo -e "  3. ${CYAN}./docker-helper.sh train-simple${NC}  # Test training"
    echo -e "  4. ${CYAN}./docker-helper.sh test core${NC}     # Validate system"
    echo -e "  5. ${CYAN}./docker-helper.sh submit${NC}        # Generate submission (menu)"
    echo
}

# Main command dispatcher
main() {
    local command=${1:-help}
    local subcommand=${2:-""}
    
    if [[ "$command" != "help" ]]; then
        show_banner
    fi
    
    case $command in
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
        "train-nested")
            train_nested
            ;;
        "train-simple")
            train_simple
            ;;
        "test")
            run_tests "$subcommand"
            ;;
        "eda")
            run_eda
            ;;
        "analysis"|"analyze")
            run_analysis
            ;;
        "debug-sparse"|"debug-sparsity")
            debug_sparsity
            ;;
        "submit"|"submission")
            generate_submission "$subcommand"
            ;;
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
        *)
            log_error "Unknown command: '$command'"
            echo
            echo -e "${WHITE}Available commands:${NC}"
            echo -e "  ./docker-helper.sh help    # Show help"
            echo -e "  ./docker-helper.sh setup   # Setup system"
            echo -e "  ./docker-helper.sh status  # System status"
            echo
            exit 1
            ;;
    esac
}

# Script execution
trap 'echo -e "\n${YELLOW}Script interrupted${NC}"; exit 130' INT
main "$@"