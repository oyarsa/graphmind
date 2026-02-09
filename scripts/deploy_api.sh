#!/bin/bash
set -e
cd /opt/graphmind
git pull
docker compose up -d --build
docker image prune -f
