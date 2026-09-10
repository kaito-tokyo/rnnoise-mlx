# Windows / Colab learning handoff

Updated: 2026-09-10 (JST)

This document hands the persistent control-plane role to
`prodesk-400-g4-dm`. The Windows machine is the durable MLflow and future
supervisor host; Colab is an ephemeral L4 execution environment. A Mac remains
an operator and development machine, not the authority for a running training
session.

## Current Windows state

The following state was read from the Windows host on 2026-09-10.

| Item | Value |
| --- | --- |
| Host | `prodesk-400-g4-dm` (accessible with `ssh prodesk-400-g4-dm`) |
| OS | Windows 11 Pro, 64-bit |
| CPU | Intel Core i3-8100T, 4 cores / 4 logical processors |
| Memory | 8 GiB installed; one 8 GiB SO-DIMM is present, one of two slots is free; firmware reports 32 GiB maximum |
| MLflow | 3.14.0, Windows-local SQLite backend and artifact store |
| MLflow task | `RNNoise-MLflow`, state `Running` |
| MLflow Python | `%LOCALAPPDATA%\\MLflow\\.venv\\Scripts\\python.exe` |
| MLflow database | `%LOCALAPPDATA%\\MLflow\\data\\mlflow.db` |
| MLflow artifacts | `%LOCALAPPDATA%\\MLflow\\artifacts` |
| Tailnet endpoint | `https://prodesk-400-g4-dm.salmon-pollux.ts.net/` |
| Tailscale Serve target | `http://127.0.0.1:5000`, tailnet only |

The MLflow server binds to Windows loopback. Tailscale Serve supplies the HTTPS
tailnet endpoint; do not expose MLflow through a public Funnel or a raw public
port.

## Verified learning and recovery path

The Colab L4 validation run is:

- experiment: `rnnoise-colab-checkpoint-validation` (ID `3`)
- run ID: `6d1c3c9aa58b4eaca2f0199b9ca22d97`
- UI: <https://prodesk-400-g4-dm.salmon-pollux.ts.net/#/experiments/3/runs/6d1c3c9aa58b4eaca2f0199b9ca22d97>

It established the following end-to-end behavior.

1. A Colab L4 process received `SIGTERM` at update 10.
2. It completed the current batch, saved its model, optimizer, PRNG, cursor, and
   history, and uploaded a new immutable checkpoint attempt to MLflow.
3. The uploader downloaded and SHA-256-verified that payload before publishing
   `complete.json` last.
4. A new process downloaded the Windows-hosted checkpoint, with the original
   Colab checkpoint deliberately removed from the recovery path.
5. It resumed the same MLflow run through update 20 using relocated feature
   data and manifests.
6. Model, optimizer, MLX random state, history, update/cursor state, and
   initial evaluation matched an uninterrupted reference run exactly.

The latest committed checkpoint is
`checkpoints/update-00000020/151f881e1b964bfead6ec2682d938814`. Its manifest
SHA-256 is
`6e763e5013beb5083072d833d88735669ac950a6ad1910d156a158b72897c92e`.

The local evidence archive is
`experiments/2026-09-10-rnnoise-l4-checkpoint-resume/validation.tar.gz`; its
SHA-256 is
`82e2851d2fa8c37e25a6f932b45464e66e392ca7da3819de3be655b27bf8b58c`.
See [the checkpoint validation record](mlflow-checkpoint-validation.md) for
the full test environment and limitations.

## Source state that must be transferred deliberately

The checkpoint implementation is committed as
`93c0914` (`Commit verified MLflow checkpoints`) on
`umireon/portable-training-storage`. The working tree may still include
unrelated pre-existing `audits/` content. A fresh Windows clone must check out
that branch and commit before it is treated as executable authority.

The commit contains the following coherent change:

- `src/rnnoise_mlx/tools/mlflow_checkpoint.py`
- `src/rnnoise_mlx/tests/test_remote_checkpoint.py`
- `src/rnnoise_mlx/training/checkpoint.py`
- `src/rnnoise_mlx/training/tracking.py`
- `src/rnnoise_mlx/training/train.py`
- `src/rnnoise_mlx/tests/test_checkpoint.py`
- `README.md`
- `docs/portable-training-storage.md`

The validation record and this handoff are documentation commits that follow
the implementation commit.

The last local test result is `185 passed`. That result is macOS/MLX based; it
does not prove the ordinary project dependency installation on Linux/Colab.
The live Colab validation used Python 3.13.15, MLX CUDA 0.32.2, MLflow 3.14.0,
and an explicit `PYTHONPATH` source snapshot. The repository currently pins
MLX below 0.32, so Linux/Colab dependency policy remains a separate task.

## Network and secret boundaries

The existing tailnet policy permits a tagged Colab node to reach this Windows
host on TCP 443 only. The intended tag is `tag:colab`; the Windows MLflow
endpoint is reached through Tailscale Serve, never by direct port 5000 access.

Do not put any of the following in this repository, MLflow parameters, logs,
or an LLM prompt:

- Tailscale auth keys
- Google / Colab OAuth material
- Windows Credential Manager exports
- full environment dumps containing secrets
- raw training corpora

For Windows-managed Colab sessions, provision a suitably restricted tagged
Tailscale auth key into Windows Credential Manager or another Windows-local
secret mechanism. The session bootstrap may materialize it in a mode-0600
temporary file inside Colab only long enough to authenticate Tailscale, then
must remove the file. The key must never be printed.

## Proposed supervisor architecture (not implemented)

Windows should host a small deterministic supervisor, started as a user-scoped
Task Scheduler task. It should run under the same Windows user that holds the
Colab CLI OAuth state and required local secrets. Do not start with a Windows
service: a service account changes the profile and secret-access boundary.

The intended decision flow is:

```text
Windows deterministic gate -> GPT-5.3-Codex-Spark -> escalation
```

### Deterministic gate

The gate is the authority for observations, and records only factual state:

- Colab session and training child-process liveness
- last heartbeat / last metric timestamp
- MLflow HTTPS availability
- latest `complete.json` checkpoint and its checksum
- a second writer attempting the same logical MLflow run
- Windows free disk space and supervisor process health

It must not require an LLM to identify these conditions. It creates a compact,
redacted `incident.json` only when a state transition requires attention.

### GPT-5.3-Codex-Spark

Spark receives the redacted incident, selected MLflow facts, and a bounded log
tail. It provides triage and an operator-oriented explanation, not authority to
change a session. Require structured output:

```json
{
  "severity": "warning | critical",
  "facts": ["observed facts only"],
  "checkpoint": "update-00000000 or null",
  "recommended_action": "resume | investigate | wait",
  "requires_human_approval": true,
  "escalate_to": "user | codex"
}
```

Normal operation never calls Spark. Spark has a separate usage limit, so the
gate must remain useful when Spark is unavailable.

### Escalation

The initial escalation action must be read-only and human-approved:

- notify the operator with evidence and the latest valid checkpoint;
- request a normal Codex task only for ambiguous incidents, source changes, or
  recovery that needs implementation work;
- never automatically spend Colab compute units by resuming a run;
- never automatically stop a run except for pre-approved deterministic safety
  rules such as a confirmed duplicate writer or storage integrity failure.

## Session semantics

A logical experiment spans many disposable Colab sessions. One logical MLflow
run has only one active writer at a time. A session ID is an append-only event,
not a replacement for the MLflow run ID.

```text
created -> preparing -> running -> stopping -> paused
                                  -> lost
running -> completed | failed
paused/lost -> resuming -> running
```

MLflow has no native `paused` run status. Store the logical state, latest
checkpoint, session ID, and stop reason as session-event artifacts/tags; do not
infer current Colab liveness from MLflow `FINISHED` or `RUNNING` alone.

Use a large update-based `--checkpoint-every` for ordinary runs. Checkpoint
transfer and verification are synchronous and each test payload was roughly
33 MiB. Graceful `SIGINT`/`SIGTERM` already requests a final checkpoint at
the next safe batch boundary, so a time-based checkpoint control is unnecessary.

## Windows setup sequence

1. Increase memory if desired. The supervisor and MLflow work at 8 GiB; 16 GiB
   is adequate for Windows plus on-demand small local inference, while a
   matching 16 GiB x 2 pair gives 32 GiB for future local-model experiments.
2. Transfer the reviewed checkpoint source change to Windows; verify the branch
   and `git status` before treating it as executable authority.
3. Install and authenticate the Colab CLI as the intended Windows user. Prove
   that the CLI can create, query, and stop a harmless session non-interactively.
4. Place the restricted Tailscale auth key in a Windows-local secret store and
   prove that a Colab bootstrap receives only `tag:colab` and reaches MLflow
   HTTPS on port 443.
5. Re-run the small checkpoint validation from Windows control, retaining the
   same properties: stop, Windows download, relocated-data resume, and
   uninterrupted-reference comparison.
6. Implement the deterministic gate and session-event journal.
7. Register the user-scoped Task Scheduler task with no overlapping instances,
   restart-on-failure, and no 72-hour execution limit.
8. Add Spark triage and a user-approved escalation destination only after the
   gate is independently useful.

## Operational commands

Read-only MLflow task and proxy checks:

```powershell
Get-ScheduledTask -TaskName RNNoise-MLflow
tailscale serve status
& "$env:LOCALAPPDATA\MLflow\.venv\Scripts\python.exe" -m mlflow --version
```

Do not use a `file:` or `sqlite:` URI for a training invocation. Training
must preflight an available HTTP(S) MLflow server before it creates output. The
existing code deliberately refuses local fallback.

## Explicit non-goals at handoff

- No Windows Colab supervisor has been implemented or installed.
- No automatic resume, automatic model selection, or automatic source editing
  is authorized.
- No local LLM is required for the first supervisor version.
- No production corpus training or quality conclusion has been made from the
  synthetic checkpoint validation.
