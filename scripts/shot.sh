#!/bin/sh
# Screenshot dist/index.html with the installed Chrome. Usage: shot.sh [light|dark] [height]
cd "$(dirname "$0")/.."
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
SCHEME="${1:-light}"
OUT="build/shot-$SCHEME.png"
rm -f "$OUT"
"$CHROME" --headless=new --disable-gpu --hide-scrollbars --no-first-run --user-data-dir="$(mktemp -d)" \
  --force-prefers-color-scheme="$SCHEME" \
  --window-size=1280,"${2:-5200}" --virtual-time-budget=8000 --enable-logging=stderr --v=0 \
  --screenshot="$OUT" "file://$PWD/dist/index.html?theme=$SCHEME" 2> build/console.log &
PID=$!
i=0
while [ ! -s "$OUT" ] && [ $i -lt 60 ]; do sleep 1; i=$((i + 1)); done
sleep 1
kill $PID 2>/dev/null
git grep --no-index -h -e "CONSOLE" -e "Uncaught" -- build/console.log | cut -c1-300
ls -la "$OUT"
