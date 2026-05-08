#!/usr/bin/env bash
# One-shot local bootstrap.
# Idempotent: safe to re-run.
# Usage:
#   bash scripts/setup_local_env.sh            # python venv + data + train + tests
#   bash scripts/setup_local_env.sh --with-docker      # also install Colima + Docker CLI
#   bash scripts/setup_local_env.sh --with-k8s         # also install Minikube + kubectl
#   bash scripts/setup_local_env.sh --all              # everything

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WITH_DOCKER=0
WITH_K8S=0
for a in "$@"; do
  case "$a" in
    --with-docker) WITH_DOCKER=1 ;;
    --with-k8s)    WITH_K8S=1 ;;
    --all)         WITH_DOCKER=1; WITH_K8S=1 ;;
    *) echo "unknown flag: $a"; exit 1 ;;
  esac
done

say() { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
ok()  { printf "    \033[1;32m\xe2\x9c\x93\033[0m %s\n" "$*"; }
warn(){ printf "    \033[1;33m!\033[0m %s\n" "$*"; }

# ---------------------------------------------------------------- 0. Python
say "Checking Python"
PY=$(command -v python3 || true)
if [ -z "$PY" ]; then
  echo "python3 not found. Install Python 3.10+ (e.g. brew install python@3.12)."
  exit 1
fi
PYV=$($PY -c 'import sys; print("%d.%d"%sys.version_info[:2])')
ok "python3 $PYV at $PY"

# ---------------------------------------------------------------- 1. venv
say "Creating .venv (Python $PYV)"
if [ ! -d .venv ]; then
  $PY -m venv .venv
  ok ".venv created"
else
  ok ".venv already present"
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --quiet --upgrade pip wheel

say "Installing Python dependencies"
pip install --quiet -r requirements.txt
ok "requirements.txt installed"

# ---------------------------------------------------------------- 2. Data
say "Preparing dataset"
if [ ! -f data/heart_disease.csv ]; then
  python scripts/download_data.py
fi
ok "dataset ready ($(wc -l < data/heart_disease.csv | tr -d ' ') rows)"

# ---------------------------------------------------------------- 3. Train
say "Training models (this also populates ./mlruns for the MLflow UI)"
if ls models/*.pkl >/dev/null 2>&1; then
  ok "models already trained ($(ls models/*.pkl | wc -l | tr -d ' ') artifacts found)"
  warn "delete models/*.pkl to force a retrain"
else
  python src/models/train.py
fi

# ---------------------------------------------------------------- 4. Tests
say "Running unit tests"
python -m pytest -q
ok "tests passed"

# ---------------------------------------------------------------- 5. Optional: Docker
# Three cases we want to distinguish, in order of preference:
#   (a) docker CLI exists AND engine responds  -> use it, skip Colima
#   (b) docker CLI exists but engine is stopped (Rancher/Docker Desktop/Podman
#       installed but not running)             -> warn, skip Colima install
#   (c) no docker CLI at all                   -> install Colima via brew
if [ "$WITH_DOCKER" = "1" ]; then
  say "Provisioning a Docker engine"

  if command -v docker >/dev/null 2>&1; then
    DOCKER_PATH=$(command -v docker)
    if docker info >/dev/null 2>&1; then
      # case (a)
      ok "docker engine is up at $DOCKER_PATH — skipping Colima install"
      docker version --format '    client {{.Client.Version}} / server {{.Server.Version}}' 2>/dev/null || true
      if docker compose version >/dev/null 2>&1; then
        ok "docker compose v2 plugin available"
      else
        warn "docker compose v2 plugin not found; run 'brew install docker-compose'"
      fi
    else
      # case (b) — never auto-install a second runtime alongside an existing one
      warn "docker CLI found at $DOCKER_PATH but the engine is not running"
      warn "start your existing engine (Rancher Desktop / Docker Desktop / 'podman machine start')"
      warn "and re-run, OR uninstall it first to let this script install Colima"
    fi
  elif ! command -v brew >/dev/null; then
    warn "no docker CLI found and Homebrew is not installed"
    warn "install Homebrew from https://brew.sh and re-run with --with-docker"
  else
    # case (c)
    say "No docker CLI detected — installing Colima (open-source, no Docker Desktop needed)"
    brew list colima         >/dev/null 2>&1 || brew install colima
    brew list docker         >/dev/null 2>&1 || brew install docker
    brew list docker-compose >/dev/null 2>&1 || brew install docker-compose
    ok "colima + docker CLI installed"
    if ! colima status >/dev/null 2>&1; then
      say "Starting Colima (4 CPUs, 4 GiB RAM) — first start takes ~60 s"
      colima start --cpu 4 --memory 4
    fi
    docker version --format '    docker {{.Client.Version}} / {{.Server.Version}}' || true
  fi
fi

# ---------------------------------------------------------------- 6. Optional: K8s
# Same three-case detection for Kubernetes:
#   (a) kubectl exists AND a cluster is reachable  -> use it, skip Minikube
#   (b) kubectl exists but no cluster responds     -> warn, skip Minikube install
#   (c) no kubectl at all                          -> install Minikube via brew
if [ "$WITH_K8S" = "1" ]; then
  say "Provisioning a Kubernetes cluster"

  if command -v kubectl >/dev/null 2>&1; then
    if kubectl cluster-info >/dev/null 2>&1; then
      # case (a)
      CTX=$(kubectl config current-context 2>/dev/null || echo "default")
      ok "kubernetes cluster reachable (context: $CTX) — skipping Minikube install"
      kubectl get nodes 2>/dev/null | sed 's/^/    /'
    else
      # case (b)
      warn "kubectl found at $(command -v kubectl) but no cluster is reachable"
      warn "enable Kubernetes in your existing tool (Rancher Desktop -> Preferences -> Kubernetes,"
      warn "or 'minikube start', or 'kind create cluster') and re-run"
    fi
  elif ! command -v brew >/dev/null; then
    warn "no kubectl found and Homebrew is not installed"
    warn "install Homebrew from https://brew.sh and re-run with --with-k8s"
  else
    # case (c)
    say "No kubectl detected — installing Minikube + kubectl"
    brew list minikube >/dev/null 2>&1 || brew install minikube
    brew list kubectl  >/dev/null 2>&1 || brew install kubectl
    ok "minikube + kubectl installed"
    if ! minikube status >/dev/null 2>&1; then
      say "Starting Minikube (driver=docker, 2 CPUs, 4 GiB)"
      minikube start --driver=docker --cpus=2 --memory=4096
    fi
    kubectl get nodes
  fi
fi

# ---------------------------------------------------------------- Summary
say "All done"
cat <<EOF

  Activate the venv:        source .venv/bin/activate
  Start the API:            python -m uvicorn src.api.app:app --reload
  Start MLflow UI:          mlflow ui --backend-store-uri ./mlruns --port 5000
  Bring up monitoring:      docker compose -f deployment/docker/docker-compose.yml up -d
                            (needs any docker engine — re-run with --with-docker
                             to install Colima only if none was detected)
  Deploy to local k8s:      kubectl apply -f deployment/kubernetes/
                            (needs any reachable cluster — re-run with --with-k8s
                             to install Minikube only if none was detected)

EOF
