#!/usr/bin/env sh
# ===========================================================================
# One-time setup for EnhancedSmartSurveillance (macOS/Linux).
# Creates ./env with Python 3.10 and installs every requirement, including
# the CPU build of PyTorch. After this, starting the app is just:
#     source env/bin/activate      # or: conda activate ./env
#     python app.py
# ===========================================================================
set -e
cd "$(dirname "$0")"

if command -v conda >/dev/null 2>&1; then
    [ -d env ] || conda create --prefix ./env python=3.10 -y
    # shellcheck disable=SC1091
    . "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate ./env
else
    [ -d env ] || python3 -m venv env
    # shellcheck disable=SC1091
    . env/bin/activate
fi

python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Setup complete. Activate the env and run: python app.py"
