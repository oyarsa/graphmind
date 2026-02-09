#!/bin/bash
set -euo pipefail
cd /opt/graphmind
git pull --quiet 1>/dev/null
docker compose up -d --build --quiet-pull 1>/dev/null
docker image prune -f 1>/dev/null
