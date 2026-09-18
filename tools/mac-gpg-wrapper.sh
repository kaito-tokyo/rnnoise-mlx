#!/usr/bin/env bash
set -euo pipefail

socket="${XDG_RUNTIME_DIR:-/tmp}/macmini-gpg-agent.sock"
gnupg_home="${HOME}/.gnupg-macmini"
mkdir -p "${gnupg_home}"
chmod 700 "${gnupg_home}"

if [[ ! -S "${socket}" ]]; then
  remote_socket="$(ssh macmini 'for p in "$(command -v gpgconf 2>/dev/null)" /opt/homebrew/bin/gpgconf /usr/local/bin/gpgconf; do if [[ -x "$p" ]]; then "$p" --list-dir agent-extra-socket; exit; fi; done; exit 127' | tail -n 1)"
  [[ -n "${remote_socket}" ]] || {
    echo "Could not determine Mac mini gpg-agent extra socket" >&2
    exit 1
  }
  ssh -fN \
    -o ExitOnForwardFailure=yes \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -L "${socket}:${remote_socket}" \
    macmini
fi

agent_socket="${gnupg_home}/S.gpg-agent"
if [[ ! -e "${agent_socket}" ]]; then
  ln -s "${socket}" "${agent_socket}"
fi

export GNUPGHOME="${gnupg_home}"
if [[ "${1-}" == "mac-gpg-wrapper" ]]; then
  shift
fi

# Keep only the public key locally.  The secret key remains in the Mac mini's
# gpg-agent; importing the public key lets the local gpg client resolve the
# signing key while forwarding the actual signing operation to that agent.
if ! gpg --batch --list-keys 208EED8B032B6537FEB9D3F8AB1412638CEE89FF >/dev/null 2>&1; then
  public_key_file="${gnupg_home}/umireon.gpg"
  curl --fail --silent --show-error --location https://github.com/umireon.gpg \
    --output "${public_key_file}"
  gpg --batch --import "${public_key_file}" >/dev/null
  rm -f "${public_key_file}"
fi

remote_gpg=""
for candidate in /opt/homebrew/bin/gpg /usr/local/bin/gpg /usr/bin/gpg; do
  if ssh macmini "test -x '${candidate}'"; then
    remote_gpg="${candidate}"
    break
  fi
done
[[ -n "${remote_gpg}" ]] || {
  echo "Could not find gpg on Mac mini" >&2
  exit 1
}

# Let the Mac mini's gpg-agent perform the signing operation.  The public key
# above is only for the local client's key resolution and verification.
exec ssh macmini "${remote_gpg}" "$@"
