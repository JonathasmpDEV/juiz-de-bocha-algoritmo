#!/usr/bin/env bash
# Linux: abre o app de benchmark do Juiz de Bocha em http://localhost:8501
# Uso:  bash iniciar.sh   (ou ./iniciar.sh depois de  chmod +x iniciar.sh)
set -e
cd "$(dirname "$0")"

UV="$(command -v uv || true)"
[ -z "$UV" ] && [ -x "$HOME/.local/bin/uv" ] && UV="$HOME/.local/bin/uv"
[ -z "$UV" ] && [ -x "$HOME/.cargo/bin/uv" ] && UV="$HOME/.cargo/bin/uv"
if [ -z "$UV" ]; then
    echo "Nao encontrei o uv. Instale com:  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

exec "$UV" run streamlit run app.py --server.port 8501 --server.headless false
