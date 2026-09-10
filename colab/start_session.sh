#!/usr/bin/env bash
# Run from WSL. Creates a Colab session and configures its userspace Tailscale.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: colab/start_session.sh --session NAME --mlflow-uri HTTPS_URL [options]

Options:
  --gpu TYPE             Request a Colab GPU, for example L4.
  --auth-key-file PATH   Defaults to ~/.config/rnnoise/tailscale-colab-authkey.
  --identity PATH        Defaults to ~/.ssh/colab_runtime_ed25519.
  --reuse                Do not create the named session first.
EOF
}

session=""
mlflow_uri=""
gpu=""
auth_key_file="$HOME/.config/rnnoise/tailscale-colab-authkey"
identity="$HOME/.ssh/colab_runtime_ed25519"
create=1
while (($#)); do
  case "$1" in
    --session) session=${2:?}; shift 2 ;;
    --mlflow-uri) mlflow_uri=${2:?}; shift 2 ;;
    --gpu) gpu=${2:?}; shift 2 ;;
    --auth-key-file) auth_key_file=${2:?}; shift 2 ;;
    --identity) identity=${2:?}; shift 2 ;;
    --reuse) create=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$session" && -n "$mlflow_uri" ]] || { usage >&2; exit 2; }
[[ -r "$auth_key_file" ]] || { echo "Cannot read auth key: $auth_key_file" >&2; exit 1; }
[[ -r "$identity" ]] || { echo "Cannot read SSH identity: $identity" >&2; exit 1; }

if (( create )); then
  args=(new --session "$session")
  [[ -n "$gpu" ]] && args+=(--gpu "$gpu")
  colab "${args[@]}"
fi

ssh_args=(
  -i "$identity"
  -o "ProxyCommand=colab ssh --proxy-mode --session $session --identity $identity"
  -o StrictHostKeyChecking=accept-new
  root@colab-runtime
)

# Auth key contents travel only through the encrypted SSH stream and exist in
# Colab tmpfs until bootstrap_tailscale.sh consumes them.
cat "$auth_key_file" | ssh "${ssh_args[@]}" \
  'umask 077; cat > /dev/shm/.tailscale-authkey; chmod 600 /dev/shm/.tailscale-authkey'

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
remote_env="MLFLOW_TRACKING_URI=$(printf '%q' "$mlflow_uri") TAILSCALE_HOSTNAME=$(printf '%q' "colab-$session")"
ssh "${ssh_args[@]}" "$remote_env bash -s" \
  < "$script_dir/bootstrap_tailscale.sh"

cat <<EOF

Ready: $session
Use the same SSH ProxyCommand and set HTTP_PROXY and HTTPS_PROXY to http://127.0.0.1:1055
inside the process that invokes MLflow. Stop the session with:
  colab stop --session $session
EOF
