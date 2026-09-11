# Colab helpers

These helpers run an ephemeral Colab GPU runtime from WSL. WSL is configured
in mirror mode, so the remote training process uses the Windows MLflow server
at `http://localhost:5000`.

`start_session.sh`:

1. creates a Colab runtime;
2. creates an SSH reverse forward from Colab `localhost:5000` to WSL
 `127.0.0.1:5000`; and
3. checks the forwarded MLflow `/health` endpoint.

The command remains attached to the remote shell so the SSH reverse forward
stays alive. Run the training command in that shell; exiting it closes the
forward. Stop the Colab session separately when finished.

Example:

```sh
./colab/start_session.sh \
  --auth adc \
  --session rnnoise-smoke \
  --gpu L4
```

The default MLflow URI is `http://localhost:5000`. The helper keeps the SSH
reverse forward alive while its remote shell is open, so remote training
processes can use that URI without a separate network service.

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
