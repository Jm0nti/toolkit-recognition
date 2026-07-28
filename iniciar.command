#!/bin/bash
# Iniciador rápido del Detector de herramientas (Mac).
# Doble clic en Finder, o desde la terminal:  ./iniciar.command
# Se para con Ctrl+C o cerrando la ventana.

cd "$(dirname "$0")" || exit 1

PY=".venv/bin/python"
if [ ! -x "$PY" ]; then
  echo "No encuentro el entorno .venv."
  echo "Créalo con:  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  read -r -p "Enter para cerrar..."
  exit 1
fi

# busca el primer puerto libre desde el 8000
PORT=""
for p in 8000 8001 8002 8003 8004; do
  if ! lsof -i :"$p" >/dev/null 2>&1; then PORT=$p; break; fi
done
if [ -z "$PORT" ]; then
  echo "No hay puertos libres entre 8000 y 8004. Cierra algún proceso e intenta de nuevo."
  read -r -p "Enter para cerrar..."
  exit 1
fi
[ "$PORT" != "8000" ] && echo "El puerto 8000 estaba ocupado, uso el $PORT."
URL="http://localhost:$PORT"
echo "Detector de herramientas  ->  $URL"

# abre el navegador apenas el server responda
( for _ in $(seq 1 30); do
    curl -s -o /dev/null "$URL" && { open "$URL"; break; }
    sleep 1
  done ) &

exec "$PY" -m uvicorn scripts.serve_app:app --host 127.0.0.1 --port "$PORT"
