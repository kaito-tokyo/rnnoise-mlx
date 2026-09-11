#!/usr/bin/env bash
# Run inside a Colab runtime.  The auth key must already be present in the
# file named by TAILSCALE_AUTHKEY_FILE; this script deletes it after `up`.
set -euo pipefail

: "${MLFLOW_TRACKING_URI:?Set MLFLOW_TRACKING_URI to the HTTPS Tailscale Serve URL}"
: "${TAILSCALE_AUTHKEY_FILE:=/dev/shm/.tailscale-authkey}"
: "${TAILSCALE_HOSTNAME:=colab-rnnoise}"
MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI%/}
[[ -n "$MLFLOW_TRACKING_URI" ]] || { echo "MLFLOW_TRACKING_URI must not be only slashes" >&2; exit 1; }

key_file="$TAILSCALE_AUTHKEY_FILE"
cleanup_key() { rm -f -- "$key_file"; }
trap cleanup_key EXIT

if [[ ! -s "$key_file" ]]; then
  echo "Tailscale auth-key file is missing or empty: $key_file" >&2
  exit 1
fi

if ! command -v tailscaled >/dev/null 2>&1; then
  curl --fail --silent --show-error https://tailscale.com/install.sh | sh
fi

if ! pgrep -x tailscaled >/dev/null 2>&1; then
  nohup tailscaled \
    --tun=userspace-networking \
    --socks5-server=127.0.0.1:1055 \
    --outbound-http-proxy-listen=127.0.0.1:1055 \
    >/tmp/tailscaled.log 2>&1 &
fi

for _ in $(seq 1 30); do
  tailscale status >/dev/null 2>&1 && break
  if ! pgrep -x tailscaled >/dev/null 2>&1; then
    cat /tmp/tailscaled.log >&2 || true
    exit 1
  fi
  sleep 1
done
tailscale status >/dev/null

tailscale up --auth-key="file:$key_file" --hostname="$TAILSCALE_HOSTNAME"

export HTTP_PROXY=http://127.0.0.1:1055
export HTTPS_PROXY=http://127.0.0.1:1055
export http_proxy="$HTTP_PROXY"
export https_proxy="$HTTPS_PROXY"
health_status=$(curl --fail --silent --show-error --max-time 20 \
  --output /dev/null --write-out '%{http_code}' "$MLFLOW_TRACKING_URI/health")
if [[ "$health_status" != 200 ]]; then
  echo "MLflow health check returned HTTP $health_status" >&2
  exit 1
fi
echo "Tailscale is ready. Set HTTP_PROXY and HTTPS_PROXY to http://127.0.0.1:1055 for MLflow clients."
