# Colab helpers

Install the local Colab CLI control-plane dependencies from
`requirements-colab-cli-lock.txt`. This is separate from the Python environment
inside the Colab VM:

```sh
pipx inject google-colab-cli -r requirements-colab-cli-lock.txt
```

The lock deliberately pins `jupyter-kernel-client==0.15.0`; newer 1.0.x
releases do not expose the `KernelClient` API expected by Colab CLI 0.7.0.
Inside the GPU VM, install the project and its CUDA MLX dependencies from
`pyproject.toml`:

```sh
python -m pip install -e /content/rnnoise-mlx
```

These helpers run an ephemeral Colab GPU runtime from WSL. The training code
is imported directly from the cloned repository, so editable installation is
intentional for interactive experiments.

`start_session.sh`:

1. creates a Colab runtime;
2. attaches an SSH session to the runtime; and
3. keeps that session alive while interactive work is in progress.

The command remains attached to the remote shell for interactive work. Run
the training command in that shell; exiting it closes the SSH session. Stop
the Colab session separately when finished.

Example:

```sh
./colab/start_session.sh \
  --auth adc \
  --session rnnoise-smoke \
  --gpu L4
```

The helper uses the Colab CLI `adc` authentication strategy by default. Set up
ADC once with:

```sh
gcloud auth application-default login
```

Use `--auth oauth2` to select the Colab CLI's direct OAuth flow instead. The
SSH private key is machine-local and must not be committed.

When the workload has stopped, release the Colab runtime:

```sh
colab --auth adc stop --session rnnoise-smoke
```
