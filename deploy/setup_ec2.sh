#!/usr/bin/env bash
# One-time EC2 provisioning for trend-guard. Run this ON the EC2 instance (Ubuntu), as a user
# with sudo. See BUILD_SPEC.md Section 13.2 and docs/RUNBOOK.md.
#
# Before running this script:
#   1. On your DEV MACHINE, generate a deploy key and add it as READ-ONLY to the private repo:
#        ssh-keygen -t ed25519 -f deploy_key -N ""
#        gh repo deploy-key add deploy_key.pub --title ec2 -R <owner>/trend-guard
#      Then copy the PRIVATE half (deploy_key) to this instance as ~/.ssh/id_ed25519 (mode 600).
#   2. Fill in /etc/trend-guard/env (created below) with real API keys AFTER this script runs --
#      never put secrets in this script or in git.

set -euo pipefail

REPO_SSH="git@github.com:REPLACE_ME/trend-guard.git"   # TODO: fill in the real owner/repo
SERVICE_USER="trendguard"
APP_DIR="/opt/trend-guard"

echo "== timezone: UTC, clock sync via chrony =="
sudo timedatectl set-timezone UTC
sudo apt-get update -y
sudo apt-get install -y chrony python3.11 python3.11-venv git logrotate
sudo systemctl enable --now chrony

echo "== service user =="
if ! id -u "$SERVICE_USER" >/dev/null 2>&1; then
  sudo useradd --system --create-home --shell /usr/sbin/nologin "$SERVICE_USER"
fi

echo "== clone (read-only deploy key must already be on this host at ~/.ssh/id_ed25519) =="
sudo mkdir -p "$APP_DIR"
sudo chown "$SERVICE_USER":"$SERVICE_USER" "$APP_DIR"
if [ ! -d "$APP_DIR/.git" ]; then
  sudo -u "$SERVICE_USER" git clone "$REPO_SSH" "$APP_DIR"
fi

echo "== python venv =="
sudo -u "$SERVICE_USER" python3.11 -m venv "$APP_DIR/.venv"
sudo -u "$SERVICE_USER" "$APP_DIR/.venv/bin/pip" install --upgrade pip
sudo -u "$SERVICE_USER" "$APP_DIR/.venv/bin/pip" install -r "$APP_DIR/requirements.txt"

echo "== secrets file (fill in manually after this script, then chmod 600) =="
sudo mkdir -p /etc/trend-guard
if [ ! -f /etc/trend-guard/env ]; then
  sudo cp "$APP_DIR/.env.example" /etc/trend-guard/env
fi
sudo chown "$SERVICE_USER":"$SERVICE_USER" /etc/trend-guard/env
sudo chmod 600 /etc/trend-guard/env
echo "  -> edit /etc/trend-guard/env now and set ROOSTOO_API_KEY / ROOSTOO_SECRET_KEY / ALLOW_LIVE_ORDERS"

echo "== HALT kill-switch file location =="
sudo touch /etc/trend-guard/HALT.example
echo "  -> to halt the bot: sudo mv /etc/trend-guard/HALT.example /etc/trend-guard/HALT (bot cancels orders and idles)"
echo "  -> to resume: sudo rm /etc/trend-guard/HALT"
sudo rm -f /etc/trend-guard/HALT.example  # created only as documentation above; not present by default

echo "== systemd units =="
sudo cp "$APP_DIR/deploy/trend-guard.service" /etc/systemd/system/trend-guard.service
sudo cp "$APP_DIR/deploy/recorder.service" /etc/systemd/system/trend-guard-recorder.service
sudo systemctl daemon-reload
sudo systemctl enable trend-guard-recorder
sudo systemctl enable trend-guard
echo "  -> start the recorder now (safe, read-only): sudo systemctl start trend-guard-recorder"
echo "  -> DO NOT start trend-guard.service until ALLOW_LIVE_ORDERS/keys are verified"

echo "== log rotation =="
sudo cp "$APP_DIR/deploy/logrotate.conf" /etc/logrotate.d/trend-guard

echo "== done. Remaining manual steps: =="
echo "  1. Edit /etc/trend-guard/env with real keys."
echo "  2. Verify with: sudo -u $SERVICE_USER $APP_DIR/.venv/bin/python scripts/smoke_readonly.py"
echo "  3. Only then: sudo systemctl start trend-guard"
