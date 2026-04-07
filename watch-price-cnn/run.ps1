<#
.SYNOPSIS
    Task runner for watch-price-cnn (Windows equivalent of Makefile)
.USAGE
    .\run.ps1 setup       # Create venv and install deps
    .\run.ps1 train       # Train with default config
    .\run.ps1 evaluate    # Evaluate best model
    .\run.ps1 gradcam     # Evaluate + Grad-CAM visualizations
    .\run.ps1 test        # Run pytest
    .\run.ps1 lint        # Lint code
    .\run.ps1 format      # Auto-format code
    .\run.ps1 clean       # Remove outputs and caches
    .\run.ps1 docker      # Build Docker image
    .\run.ps1 docker-train # Train inside Docker with GPU
    .\run.ps1 help        # Show this help
#>

param(
    [Parameter(Position=0)]
    [string]$Task = "help",

    [string]$Config = "configs\base.yaml"
)

$ErrorActionPreference = "Stop"
$VENV = ".\.venv\Scripts"

function Invoke-Setup {
    Write-Host "`n=== Setting up environment ===" -ForegroundColor Cyan
    python -m venv .venv
    & "$VENV\python.exe" -m pip install --upgrade pip
    & "$VENV\pip.exe" install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    & "$VENV\pip.exe" install -e ".[dev]"
    & "$VENV\python.exe" -c "import torch; print(f'PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()}')"
    Write-Host "`nSetup complete. Activate with: .venv\Scripts\activate" -ForegroundColor Green
}

function Invoke-Train {
    Write-Host "`n=== Training ===" -ForegroundColor Cyan
    & "$VENV\python.exe" scripts\train.py --config $Config
}

function Invoke-Evaluate {
    Write-Host "`n=== Evaluating ===" -ForegroundColor Cyan
    & "$VENV\python.exe" scripts\evaluate.py --config $Config
}

function Invoke-GradCAM {
    Write-Host "`n=== Evaluating + Grad-CAM ===" -ForegroundColor Cyan
    & "$VENV\python.exe" scripts\evaluate.py --config $Config --gradcam
}

function Invoke-Test {
    Write-Host "`n=== Running tests ===" -ForegroundColor Cyan
    & "$VENV\python.exe" -m pytest tests\ -v
}

function Invoke-Lint {
    Write-Host "`n=== Linting ===" -ForegroundColor Cyan
    & "$VENV\python.exe" -m ruff check src\ scripts\ tests\
    & "$VENV\python.exe" -m black --check src\ scripts\ tests\
}

function Invoke-Format {
    Write-Host "`n=== Formatting ===" -ForegroundColor Cyan
    & "$VENV\python.exe" -m ruff check --fix src\ scripts\ tests\
    & "$VENV\python.exe" -m black src\ scripts\ tests\
}

function Invoke-Clean {
    Write-Host "`n=== Cleaning ===" -ForegroundColor Cyan
    $dirs = @(
        "outputs\checkpoints\*",
        "outputs\logs\*",
        "outputs\results\*"
    )
    foreach ($d in $dirs) {
        if (Test-Path $d) { Remove-Item $d -Recurse -Force }
    }
    Get-ChildItem -Recurse -Directory -Name "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Directory -Name ".ipynb_checkpoints" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "Cleaned outputs and caches" -ForegroundColor Green
}

function Invoke-Docker {
    Write-Host "`n=== Building Docker image ===" -ForegroundColor Cyan
    docker build -f docker\Dockerfile -t watch-price-cnn:latest .
}

function Invoke-DockerTrain {
    Invoke-Docker
    Write-Host "`n=== Training in Docker with GPU ===" -ForegroundColor Cyan
    docker compose -f docker\docker-compose.yaml run --rm train
}

function Show-Help {
    Write-Host "`n  watch-price-cnn task runner" -ForegroundColor Cyan
    Write-Host "  =========================="
    Write-Host ""
    Write-Host "  .\run.ps1 setup         " -NoNewline -ForegroundColor Yellow; Write-Host "Create venv + install deps (PyTorch CUDA)"
    Write-Host "  .\run.ps1 train         " -NoNewline -ForegroundColor Yellow; Write-Host "Train model"
    Write-Host "  .\run.ps1 evaluate      " -NoNewline -ForegroundColor Yellow; Write-Host "Evaluate best model"
    Write-Host "  .\run.ps1 gradcam       " -NoNewline -ForegroundColor Yellow; Write-Host "Evaluate + Grad-CAM"
    Write-Host "  .\run.ps1 test          " -NoNewline -ForegroundColor Yellow; Write-Host "Run pytest suite"
    Write-Host "  .\run.ps1 lint          " -NoNewline -ForegroundColor Yellow; Write-Host "Lint code"
    Write-Host "  .\run.ps1 format        " -NoNewline -ForegroundColor Yellow; Write-Host "Auto-format code"
    Write-Host "  .\run.ps1 clean         " -NoNewline -ForegroundColor Yellow; Write-Host "Remove outputs + caches"
    Write-Host "  .\run.ps1 docker        " -NoNewline -ForegroundColor Yellow; Write-Host "Build Docker image"
    Write-Host "  .\run.ps1 docker-train  " -NoNewline -ForegroundColor Yellow; Write-Host "Train in Docker with GPU"
    Write-Host ""
    Write-Host "  Options:"
    Write-Host "  -Config <path>          " -NoNewline -ForegroundColor Yellow; Write-Host "Config file (default: configs\base.yaml)"
    Write-Host ""
    Write-Host "  Example: .\run.ps1 train -Config configs\experiment_large.yaml"
    Write-Host ""
}

switch ($Task) {
    "setup"        { Invoke-Setup }
    "train"        { Invoke-Train }
    "evaluate"     { Invoke-Evaluate }
    "gradcam"      { Invoke-GradCAM }
    "test"         { Invoke-Test }
    "lint"         { Invoke-Lint }
    "format"       { Invoke-Format }
    "clean"        { Invoke-Clean }
    "docker"       { Invoke-Docker }
    "docker-train" { Invoke-DockerTrain }
    "help"         { Show-Help }
    default        { Write-Host "Unknown task: $Task" -ForegroundColor Red; Show-Help }
}
