# Colab helpers

These scripts connect an ephemeral **Colab** runtime to MLflow on **Windows**
through Tailscale. They are run from **WSL**, where the Colab CLI OAuth state,
the dedicated SSH key, and the Tailscale auth key are kept locally.

`start_session.sh` performs the setup that is otherwise easy to repeat
incorrectly:

1. creates a Colab runtime;
2. transfers the auth key over the SSH stream into Colab tmpfs;
3. starts Tailscale in userspace networking mode; and
4. checks the Windows MLflow `/health` endpoint through Tailscale Serve.

For example, from the repository root in WSL:

```sh
./colab/start_session.sh \
  --auth adc \
  --session rnnoise-smoke \
  --gpu L4 \
  --mlflow-uri 'https://YOUR-WINDOWS-HOST.tailnet.ts.net/'
```

The key defaults to
`~/.config/rnnoise/tailscale-colab-authkey`; override it with
`--auth-key-file` when needed. It must be a restricted reusable, ephemeral,
pre-approved key for `tag:colab`. The key, OAuth token, and SSH private key are
machine-local secrets and must never be committed.

The helper uses the Colab CLI `adc` authentication strategy by default. Set up
ADC once with `gcloud auth application-default login`. Use `--auth oauth2` to
select the Colab CLI's direct OAuth flow instead.

Colab lacks the TUN device needed for kernel networking. Consequently,
Tailscale runs in userspace mode and MLflow clients must use its local HTTP
proxy:

```sh
export HTTP_PROXY=http://127.0.0.1:1055
export HTTPS_PROXY=http://127.0.0.1:1055
```

Set those variables in the remote training process before passing the HTTPS
MLflow tracking URI. They are not needed by Windows or WSL when each accesses
its local MLflow endpoint.

When the workload has stopped, release the Colab runtime:

```sh
colab stop --session rnnoise-smoke
```

If a runtime must be released urgently, first run `tailscale logout` through
the SSH connection, then stop the session. Ephemeral Tailscale nodes also age
out after the runtime disappears.
