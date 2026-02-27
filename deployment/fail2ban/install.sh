#!/bin/bash
# Install and configure fail2ban for Caddy on the host.
# Run this script as root (or with sudo) on the deployment server.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# 1. Install fail2ban
if ! command -v fail2ban-server &>/dev/null; then
    echo "Installing fail2ban..."
    apt-get update -qq
    apt-get install -y -qq fail2ban
else
    echo "fail2ban is already installed."
fi

# 2. Resolve the actual Caddy data volume path
VOLUME_PATH=$(docker volume inspect deployment_caddy_data \
    --format '{{ .Mountpoint }}' 2>/dev/null || true)

if [ -z "$VOLUME_PATH" ]; then
    echo "ERROR: Could not find Docker volume 'deployment_caddy_data'."
    echo "Make sure the Caddy container has been started at least once."
    exit 1
fi

LOG_PATH="${VOLUME_PATH}/access.log"
echo "Caddy log path: ${LOG_PATH}"

# 3. Copy filter
echo "Installing filter to /etc/fail2ban/filter.d/caddy-status.conf"
cp "${SCRIPT_DIR}/caddy-status.conf" /etc/fail2ban/filter.d/caddy-status.conf

# 4. Copy jail (with resolved log path)
echo "Installing jail to /etc/fail2ban/jail.d/caddy.conf"
sed "s|logpath  = .*|logpath  = ${LOG_PATH}|" \
    "${SCRIPT_DIR}/caddy.conf" > /etc/fail2ban/jail.d/caddy.conf

# 5. Restart fail2ban
echo "Restarting fail2ban..."
systemctl enable fail2ban
systemctl restart fail2ban

# 6. Verify — wait for the socket to appear
echo ""
echo "Waiting for fail2ban to start..."
for i in $(seq 1 10); do
    if [ -S /var/run/fail2ban/fail2ban.sock ]; then
        break
    fi
    sleep 1
done

echo "=== Status ==="
fail2ban-client status caddy-status
