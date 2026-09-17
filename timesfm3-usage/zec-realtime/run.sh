#!/usr/bin/env bash
# Starts the dashboard on http://127.0.0.1:8000
#
# Override anything with the usual flags, e.g.
#   ./run.sh --device cuda --horizon 24 --port 9000
set -euo pipefail
cd "$(dirname "$0")"
exec python -m zecfm "$@"
