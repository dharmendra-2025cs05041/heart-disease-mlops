#!/usr/bin/env bash
# Deploy heart-disease-api to Hugging Face Spaces.
#
# Reads HF_TOKEN from environment (or sources ~/.secrets/.env if HF_TOKEN unset).
# Stages required files (src/, models/, HF Dockerfile, lean requirements, README)
# into a temporary clone of the HF Space repo, commits and pushes.
#
# Nothing persists on the local machine: the temp clone is wiped at the end,
# the user's main repo is untouched, and HF_TOKEN is only used in the push URL.

set -euo pipefail

HF_USER="dharmendra-2025cs05041"
HF_SPACE="heart-disease-api"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HF_DIR="${REPO_ROOT}/deployment/huggingface"
WORKDIR="$(mktemp -d -t hf-deploy-XXXXXX)"
trap 'rm -rf "$WORKDIR"' EXIT

# Pull HF_TOKEN from ~/.secrets/.env if not exported
if [[ -z "${HF_TOKEN:-}" ]]; then
    if [[ -r "${HOME}/.secrets/.env" ]]; then
        # shellcheck disable=SC1091
        set -a; source "${HOME}/.secrets/.env"; set +a
    fi
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set and not found in ~/.secrets/.env" >&2
    exit 1
fi

SPACE_REMOTE="https://${HF_USER}:${HF_TOKEN}@huggingface.co/spaces/${HF_USER}/${HF_SPACE}"

echo "==> Cloning HF Space into ${WORKDIR}/space"
git clone --quiet "${SPACE_REMOTE}" "${WORKDIR}/space"
cd "${WORKDIR}/space"

# Detect the default branch HF uses (usually 'main')
DEFAULT_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo main)"
echo "    default branch: ${DEFAULT_BRANCH}"

echo "==> Staging application code from ${REPO_ROOT}"
mkdir -p src/api src/models src/data_processing models

# Source files (only what's needed for serving).
# data_processing/ is required because preprocessor.pkl was pickled as an
# instance of the custom DataPreprocessor class and joblib must be able to
# import that class to unpickle it.
cp "${REPO_ROOT}/src/__init__.py"                       src/__init__.py
cp "${REPO_ROOT}/src/api/__init__.py"                   src/api/__init__.py
cp "${REPO_ROOT}/src/api/app.py"                        src/api/app.py
cp "${REPO_ROOT}/src/models/__init__.py"                src/models/__init__.py
cp "${REPO_ROOT}/src/models/predictor.py"               src/models/predictor.py
cp "${REPO_ROOT}/src/data_processing/__init__.py"       src/data_processing/__init__.py
cp "${REPO_ROOT}/src/data_processing/preprocessor.py"   src/data_processing/preprocessor.py

# Model artefacts
cp "${REPO_ROOT}/models/final_model.pkl"     models/final_model.pkl
cp "${REPO_ROOT}/models/preprocessor.pkl"    models/preprocessor.pkl

# HF-specific top-level files
cp "${HF_DIR}/Dockerfile"                    Dockerfile
cp "${HF_DIR}/requirements.txt"              requirements.txt
cp "${HF_DIR}/README.md"                     README.md

echo "==> Files staged in HF Space clone:"
git add -A
git status --short | sed 's/^/    /'

if git diff --cached --quiet; then
    echo "==> No changes vs HF Space remote — nothing to push."
    exit 0
fi

COMMIT_MSG="Deploy heart-disease-api ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
git \
    -c user.name="Dharmendra Parsaila" \
    -c user.email="282936014+dharmendra-2025cs05041@users.noreply.github.com" \
    commit --quiet -m "${COMMIT_MSG}"

echo "==> Pushing to HF Space (branch: ${DEFAULT_BRANCH})"
git push --quiet origin "${DEFAULT_BRANCH}"

echo ""
echo "==> Push successful."
echo ""
echo "    Build logs : https://huggingface.co/spaces/${HF_USER}/${HF_SPACE}"
echo "    Live URL   : https://${HF_USER}-${HF_SPACE}.hf.space"
echo "    Swagger UI : https://${HF_USER}-${HF_SPACE}.hf.space/docs"
echo ""
echo "    First build typically takes 4-7 minutes (downloading sklearn, etc.)."
echo "    Watch the 'Building' badge on the Space page until it turns 'Running'."
