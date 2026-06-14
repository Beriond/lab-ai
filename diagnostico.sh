#!/usr/bin/env bash
# Diagnóstico rápido quando o terminal trava ou scripts Python não respondem.
# Uso:  ./diagnostico.sh

set -u

OLLAMA_URL="http://localhost:11434"

echo "==================================================================="
echo "  DIAGNÓSTICO — lab-ai"
echo "==================================================================="

echo ""
echo "[1] Processos python em execução"
echo "-------------------------------------------------------------------"
PS_OUT=$(ps axo pid,stat,rss,etime,command | awk 'NR==1 || (/python/ && !/awk/)')
echo "$PS_OUT"
echo ""
# Aviso para processos suspensos (Ctrl+Z) ou zumbis
SUSPECT=$(echo "$PS_OUT" | awk 'NR>1 && ($2 ~ /^T/ || $2 ~ /^Z/) {print $1" (STAT="$2")"}')
if [ -n "$SUSPECT" ]; then
  echo "  >>> ATENÇÃO: processo(s) suspenso(s)/zumbi(s) detectado(s):"
  echo "$SUSPECT" | sed 's/^/      /'
  echo "      Mate com:  kill -9 <PID>"
fi

echo ""
echo "[2] Ollama está respondendo?"
echo "-------------------------------------------------------------------"
if curl -sS --max-time 3 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
  echo "  OK — $OLLAMA_URL respondeu"
  curl -sS --max-time 3 "$OLLAMA_URL/api/tags" \
    | tr ',' '\n' | grep '"name"' | sed 's/.*"name":"/      modelo: /; s/".*//'
else
  echo "  FALHOU — Ollama não respondeu em 3s"
  echo "      Tente:  open -a Ollama   (ou reinicie o app Ollama)"
fi

echo ""
echo "[3] Pressão de memória"
echo "-------------------------------------------------------------------"
PAGES_FREE=$(memory_pressure 2>/dev/null | awk '/Pages free/ {print $3; exit}')
if [ -n "$PAGES_FREE" ]; then
  MB_FREE=$(( PAGES_FREE * 16 / 1024 ))
  echo "  Pages free: $PAGES_FREE  (~${MB_FREE} MB livres)"
  if [ "$PAGES_FREE" -lt 5000 ]; then
    echo "  >>> CRÍTICO: menos de 80 MB livres. Sistema vai usar swap."
    echo "      Feche abas do Chrome / apps que não está usando."
  elif [ "$PAGES_FREE" -lt 15000 ]; then
    echo "  >>> APERTADO: scripts grandes podem ficar lentos."
  else
    echo "  OK"
  fi
fi

echo ""
echo "[4] Top 5 consumidores de RAM"
echo "-------------------------------------------------------------------"
ps -axo rss,comm | sort -rn | head -5 \
  | awk '{ mb=$1/1024; $1=""; printf "  %6.0f MB  %s\n", mb, $0 }'

echo ""
echo "[5] Python em uso"
echo "-------------------------------------------------------------------"
VENV_PY="/Users/carolina.zambelli/Desktop/lab-ai/.venv/bin/python"
echo "  which python : $(command -v python 2>/dev/null || echo '(nenhum)')"
echo "  venv python  : $VENV_PY"
if [ -x "$VENV_PY" ]; then
  echo "  venv version : $($VENV_PY --version 2>&1)"
else
  echo "  >>> venv não encontrado — rode:  python3 -m venv .venv"
fi

echo ""
echo "==================================================================="
echo "  Próximos passos sugeridos:"
echo "  - Se há processo com STAT=T  →  kill -9 <PID>"
echo "  - Se Ollama falhou           →  reabra o app Ollama"
echo "  - Se memória < 80MB livres   →  feche Chrome/apps"
echo "  - Para rodar scripts:        →  source .venv/bin/activate"
echo "==================================================================="
