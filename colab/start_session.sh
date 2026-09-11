#!/usr/bin/env bash
# Run from WSL. Creates a Colab session and configures its userspace Tailscale.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: colab/start_session.sh --session NAME --mlflow-uri HTTPS_URL [options]

Options:
  --gpu TYPE             Request a Colab GPU, for example L4.
  --auth TYPE             Colab auth strategy: adc or oauth2 (default: adc).
  --auth-key-file PATH   Defaults to ~/.config/rnnoise/tailscale-colab-authkey.
  --identity PATH        Defaults to ~/.ssh/colab_runtime_ed25519.
  --reuse                Do not create the named session first.
EOF
}

session=""
mlflow_uri=""
gpu=""
auth="adc"
auth_key_file="$HOME/.config/rnnoise/tailscale-colab-authkey"
identity="$HOME/.ssh/colab_runtime_ed25519"
create=1
stop_on_failure=0
while (($#)); do
  case "$1" in
    --session) session=${2:?}; shift 2 ;;
    --mlflow-uri) mlflow_uri=${2:?}; shift 2 ;;
    --gpu) gpu=${2:?}; shift 2 ;;
    --auth) auth=${2:?}; shift 2 ;;
    --auth-key-file) auth_key_file=${2:?}; shift 2 ;;
    --identity) identity=${2:?}; shift 2 ;;
    --reuse) create=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$session" && -n "$mlflow_uri" ]] || { usage >&2; exit 2; }
case "$auth" in
  adc|oauth2) ;;
  *) echo "Unsupported Colab auth strategy: $auth" >&2; exit 2 ;;
esac
mlflow_uri=${mlflow_uri%/}
[[ -n "$mlflow_uri" ]] || { echo "MLflow URI must not be only slashes" >&2; exit 2; }
[[ -r "$auth_key_file" ]] || { echo "Cannot read auth key: $auth_key_file" >&2; exit 1; }
[[ -r "$identity" ]] || { echo "Cannot read SSH identity: $identity" >&2; exit 1; }

if (( create )); then
  args=(--auth "$auth" new --session "$session")
  [[ -n "$gpu" ]] && args+=(--gpu "$gpu")
  colab "${args[@]}"
  stop_on_failure=1
  cleanup_session() {
    local status=$?
    if (( stop_on_failure )); then
      colab --auth "$auth" stop --session "$session" >&2 || true
    fi
    exit "$status"
  }
  trap cleanup_session EXIT
fi

known_hosts_dir="$HOME/.cache/rnnoise/colab-known-hosts"
mkdir -p -m 700 "$known_hosts_dir"
known_hosts="$known_hosts_dir/$session"
# A newly created Colab runtime is expected to have a new host key, even when
# its session name was used before. Reused sessions retain their prior key.
if (( create )); then
  rm -f -- "$known_hosts"
fi
proxy_command="colab --auth $(printf '%q' "$auth") ssh --proxy-mode --session $(printf '%q' "$session") --identity $(printf '%q' "$identity")"
ssh_args=(
  -i "$identity"
  -o "ProxyCommand=$proxy_command"
  -o "UserKnownHostsFile=$known_hosts"
  -o "HostKeyAlias=colab-$session"
  -o StrictHostKeyChecking=accept-new
  root@colab-runtime
)

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
remote_env="MLFLOW_TRACKING_URI=$(printf '%q' "$mlflow_uri") TAILSCALE_HOSTNAME=$(printf '%q' "colab-$session")"
# Send the key and bootstrap script through one SSH stream. The remote command
# reads the byte-counted key first, then gives the remaining stream to bash.
# Its trap also covers a bootstrap failure before bootstrap_tailscale.sh starts.
key_bytes=$(wc -c < "$auth_key_file")
remote_command='read -r key_bytes
case "$key_bytes" in (*[!0-9]*|"") exit 2;; esac
umask 077
key_file=/dev/shm/.tailscale-authkey
cleanup_key() { rm -f -- "$key_file"; }
trap cleanup_key EXIT HUP INT TERM
dd bs=1 count="$key_bytes" of="$key_file" status=none
chmod 600 "$key_file"
bash'
{
  printf '%s\n' "$key_bytes"
  cat "$auth_key_file"
  cat "$script_dir/bootstrap_tailscale.sh"
} | ssh "${ssh_args[@]}" "$remote_env bash -c $(printf '%q' "$remote_command")"

stop_on_failure=0
trap - EXIT

cat <<EOF

Ready: $session
Use the same SSH ProxyCommand and set HTTP_PROXY and HTTPS_PROXY to http://127.0.0.1:1055
inside the process that invokes MLflow. Stop the session with:
  colab --auth $auth stop --session $session
EOF
