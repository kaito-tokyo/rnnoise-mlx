#!/usr/bin/env bash
# Run from WSL. Creates a Colab session and opens an SSH shell.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: colab/start_session.sh --session NAME [options]

Options:
  --gpu TYPE              Request a Colab GPU, for example L4.
  --auth TYPE             Colab auth strategy: adc or oauth2 (default: adc).
  --identity PATH         Defaults to ~/.ssh/colab_runtime_ed25519.
  --reuse                 Do not create the named session first.
EOF
}

session=""
gpu=""
auth="adc"
identity="$HOME/.ssh/colab_runtime_ed25519"
create=1
while (($#)); do
  case "$1" in
    --session) session=${2:?}; shift 2 ;;
    --gpu) gpu=${2:?}; shift 2 ;;
    --auth) auth=${2:?}; shift 2 ;;
    --identity) identity=${2:?}; shift 2 ;;
    --reuse) create=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$session" ]] || { usage >&2; exit 2; }
case "$auth" in
  adc|oauth2) ;;
  *) echo "Unsupported Colab auth strategy: $auth" >&2; exit 2 ;;
esac
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
if (( create )); then
  rm -f -- "$known_hosts"
fi
proxy_command="colab --auth $(printf '%q' "$auth") ssh --proxy-mode --session $(printf '%q' "$session") --identity $(printf '%q' "$identity")"
ssh_options=(
  -i "$identity"
  -o "ProxyCommand=$proxy_command"
  -o "UserKnownHostsFile=$known_hosts"
  -o "HostKeyAlias=colab-$session"
  -o ExitOnForwardFailure=yes
  -o StrictHostKeyChecking=accept-new
)
ssh_host=root@colab-runtime
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  "${ssh_options[@]}" "$ssh_host" \
  'exec bash -l'

stop_on_failure=0
trap - EXIT
