#!/usr/bin/env bash
# macOS: dois cliques no Finder abre o app de benchmark do Juiz de Bocha em http://localhost:8501
# Na primeira vez:  chmod +x iniciar.command   (e, se o macOS bloquear, botão direito → Abrir)
cd "$(dirname "$0")" || exit 1

UV="$(command -v uv || true)"
for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" /opt/homebrew/bin/uv /usr/local/bin/uv; do
    [ -z "$UV" ] && [ -x "$c" ] && UV="$c"
done
if [ -z "$UV" ]; then
    echo "Nao encontrei o uv. Instale com:  brew install uv   (ou  curl -LsSf https://astral.sh/uv/install.sh | sh)"
    read -r -p "Enter para fechar..." _
    exit 1
fi

"$UV" run streamlit run app.py --server.port 8501 --server.headless false
