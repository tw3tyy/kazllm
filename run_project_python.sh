#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
for candidate in .venv/bin/python python3 /usr/local/bin/python3 /opt/homebrew/bin/python3; do
  if "$candidate" -c 'import pandas, numpy, scipy, sacrebleu, certifi' >/dev/null 2>&1; then
    exec "$candidate" "$@"
  fi
done
echo 'Python dependencies are missing. Follow the installation steps in README.md.' >&2
exit 1
