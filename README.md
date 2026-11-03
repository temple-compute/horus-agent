# horus-agent

A Horus target for machines the orchestrator cannot connect to, for example an HPC login node or a lab server behind a VPN.

The machine runs `horus-agent`. It connects **out** to a broker over HTTPS, polls it for work and runs it locally. Nothing listens on the machine, so no inbound port or VPN access is needed.

The target, kind `agent`, works like the SSH target. Each channel primitive (`launch`, `poll`, `put_file`, `get_file`, `mkdir`, ...) becomes an op that the broker queues for the agent, and the agent runs it with Horus's own `LocalTarget`. These work on agent targets exactly as they do on SSH:

- executors (shell, containers, Python environments);
- Slurm with `inner: {kind: agent, ...}`;
- the resource monitor;
- transfers.

## Install and register

The machine needs Python 3.13 or newer. `uv` downloads it if it is missing.

```sh
uv tool install horus-agent
horus-agent register --url https://api.example.com/api/v1 --token <one-time token>
horus-agent run
```

`register` stores the agent's credential in `~/.config/horus-agent/agent.json` (mode 0600). To use another location, set `HORUS_AGENT_CONFIG`. The agent's name comes from the token when the admin set one in the UI, else from `--name`, else the hostname.

`--work-dir PATH` sets where tasks run on this machine; the default is `~/horus-agent`. It is created if missing, stored in the config, and reported to the broker, so workflows do not set it. `horus-agent run --work-dir PATH` overrides it for one run.

### Keep it running

As a systemd user service (see `examples/horus-agent.service`):

```sh
cp examples/horus-agent.service ~/.config/systemd/user/
systemctl --user enable --now horus-agent
loginctl enable-linger "$USER"   # keep it running after logout
```

Without systemd (for example on a login node), run it in `tmux` or with `nohup horus-agent run &`.

Jobs run detached, in their own session, with their output in log files. Stopping, restarting or upgrading the agent does not stop them; the orchestrator picks them up again once the agent is back.

Proxies: the agent honours `HTTPS_PROXY`, `NO_PROXY` and `SSL_CERT_FILE`.

### Debug logs

`horus-agent run --debug` (or `HORUS_AGENT_DEBUG=1`) logs everything the agent does:
- every TCP and TLS connection and HTTP request;
- each poll, with how long it waited and how many ops it got;
- each op with its arguments, its duration and its result;
- every command it runs, with a preview of the output;
- file transfers, job pids, cancels, and results delivered.

Secrets stay out of the log, so it can be shared. Environment variables appear by name only, because they carry task secrets. Presigned URLs lose their query strings, which hold the signatures. Tokens are never logged.

## Use it in a workflow

```yaml
target:
  kind: agent
  agent: lab-server          # the agent's name
```

Slurm on the agent's machine:

```yaml
target:
  kind: slurm
  inner:
    kind: agent
    agent: hpc-login
```

| Field | Default | Meaning |
|---|---|---|
| `agent` | required | Agent name |
| `working_directory` | the agent's `--work-dir` | Usually unset. Absolute: used instead of the agent's directory. Relative: taken under it |
| `offline_timeout` | `900` | Seconds without a poll before the agent's tasks count as lost |

## Orchestrator configuration

The broker is set on the orchestrator, not in workflows:

| Variable | Meaning |
|---|---|
| `HORUS_AGENT_BROKER_URL` | Broker base URL |
| `HORUS_AGENT_BROKER_TOKEN` | Token for the broker |
| `HORUS_AGENT_BROKER_TOKEN_CONTEXT_KEY` | Instead of a fixed token: the `HorusContext.data` key holding a per-run token |

The broker HTTP API is described in `src/horus_agent/protocol.py`. Temple Compute OS implements it.

## Limits

- Files over 4 MiB, in either direction, go through presigned object-storage URLs that the broker issues.
- Output from `run_command(detach=False)` is buffered and capped at 4 MiB. Long-running commands go through `launch`, the default for this target.
- The `python_function_external` executor is refused: it would load bytes written on the agent's machine as code in the orchestrator.
- Task commands run as the agent's Unix user and can read its config file. Run the agent as a dedicated user, the same advice as for GitHub self-hosted runners.

## Development

```sh
uv sync --group dev
uv run make lint
uv run make test
```

`tests/integration/test_slurm.py` runs Slurm through an agent against a fake Slurm CLI (`tests/fake_slurm`) and is part of `make test`. To check it against a real cluster, start horus-slurm's docker cluster and run the opt-in test, which installs the agent in the login container:

```sh
# from a horus-slurm checkout
ITEST_DIR=/tmp/horus-agent-slurm-itest-$(id -u) \
ITEST_PROJECT=horus-agent-slurm-itest tests/integration/cluster/up.sh
# from this repo
HORUS_AGENT_SLURM_ITEST=1 uv run pytest --no-cov tests/integration/test_slurm_cluster.py
```
