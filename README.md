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
uv tool install horus-agent      # or: pip install horus-agent (also in a conda env)
horus-agent register --url https://api.example.com/api/v1
horus-agent run
```

`register` asks for the one-time token. It also reads it from stdin (`horus-agent register --url ... < token.txt`) or from `HORUS_AGENT_TOKEN`, so it never appears in the process list or shell history.

`register` stores the agent's credential in `~/.config/horus-agent/agent.json` (mode 0600; `run` refuses a config others can read). To use another location, set `HORUS_AGENT_CONFIG`. The agent's name comes from the token when the admin set one in the UI, else from `--name`, else the hostname.

`--work-dir PATH` sets where tasks run on this machine; the default is `~/horus-agent`. It is created if missing, stored in the config, and reported to the broker, so workflows do not set it. `horus-agent run --work-dir PATH` overrides it for one run.

| `register` option | Config key | Meaning |
|---|---|---|
| `--url` | `url` | Broker URL. Must be `https://` (plain `http://` only to `localhost`) |
| `--work-dir` | `work_dir` | Where tasks run |
| `--run-as USER` | `run_as` | Run task commands as this Unix user (see below) |
| `--run-as-via` | `run_as_via` | `sudo` (default) or `setpriv` |
| `--ca-bundle PEM` | `ca_bundle` | Trust only these CAs for the broker. Default: the certifi bundle |
| `--job-env NAME` | `job_env` | Pass this variable on to tasks (repeatable) |

To change a setting later, edit `agent.json` and restart the agent.

### Pinned install

Each GitHub release has a `requirements.txt` that pins horus-agent and every dependency by hash:

```sh
python3.13 -m venv /opt/horus-agent      # or: conda create -p /opt/horus-agent python=3.13 pip
/opt/horus-agent/bin/pip install --require-hashes -r requirements.txt
```

### Defaults

- The agent refuses to run as root; `--allow-root` overrides it, and tasks then run as root too.
- The broker's certificate is checked against the certifi CAs or `ca_bundle`. `SSL_CERT_FILE` and `SSL_CERT_DIR` do not apply to the broker.
- Files the agent and its tasks create are private to the agent's user (umask 077), and the work dir loses any access for other users.
- Tasks get a minimal environment: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `LANG`, `LANGUAGE`, `LC_*`, `TZ`, `TERM`, `TMPDIR`, `SSL_CERT_FILE`, `SSL_CERT_DIR` and the proxy variables, plus `job_env` and the variables the workflow sets.
- `poll` and `signal` act only on jobs the agent launched.

### Run tasks as another user

Without `run_as`, task commands run as the agent's user and can read its token. On a shared or production machine, give tasks their own user. The agent and the task user share the work dir through the task user's group:

```sh
useradd --system --home-dir /var/lib/horus-agent --shell /usr/sbin/nologin horus-agent
useradd --system --home-dir /srv/horus-agent --no-create-home --shell /usr/sbin/nologin horus-job
usermod -aG horus-job horus-agent
install -d -o horus-agent -g horus-agent -m 700 /var/lib/horus-agent
install -d -o horus-agent -g horus-job -m 2770 /srv/horus-agent
```

With `run_as` set, files are shared with that group (umask 007), never with other users. The token stays in `/var/lib/horus-agent`, which the task user cannot read.

The agent switches user in one of two ways:

- `setpriv` (util-linux): needs `CAP_SETUID` and `CAP_SETGID`, which the system unit below grants, and drops them for the task. Use this with the hardened unit.
- `sudo`: needs this rule, in `/etc/sudoers.d/horus-agent`. It does not work under `NoNewPrivileges=yes`, so use it only outside the hardened unit.

  ```
  horus-agent ALL=(horus-job) NOPASSWD: ALL
  ```

Task env values pass through the environment in both cases, never on a command line.

Do not add the agent or task user to the `docker` group: that is root on the machine. Use rootless Podman or Apptainer for container tasks.

### Run it as a system service

`examples/horus-agent.service` is a hardened system unit: `NoNewPrivileges`, `ProtectSystem=strict` with only `/srv/horus-agent` writable, a private `/tmp`, read-only home directories, and no capabilities. Tasks inherit all of it. With the users above:

```sh
python3.13 -m venv /opt/horus-agent
/opt/horus-agent/bin/pip install --require-hashes -r requirements.txt
sudo -u horus-agent env HORUS_AGENT_CONFIG=/var/lib/horus-agent/agent.json \
    /opt/horus-agent/bin/horus-agent register --url https://api.example.com/api/v1 \
    --work-dir /srv/horus-agent --run-as horus-job --run-as-via setpriv
cp examples/horus-agent.service /etc/systemd/system/
# for --run-as: switch the unit to the two CAP_SETUID/CAP_SETGID lines
systemctl enable --now horus-agent
```

Without `--run-as`, create `/srv/horus-agent` with `-g horus-agent -m 700` and skip the `horus-job` user.

### Keep it running without root

As a systemd user service (see `examples/horus-agent-user.service`):

```sh
cp examples/horus-agent-user.service ~/.config/systemd/user/horus-agent.service
systemctl --user enable --now horus-agent
loginctl enable-linger "$USER"   # keep it running after logout
```

Without systemd (for example on a login node), run it in `tmux` or with `nohup horus-agent run &`.

Jobs run detached, in their own session, with their output in log files. Stopping, restarting or upgrading the agent does not stop them; the orchestrator picks them up again once the agent is back.

Proxies: the agent honours `HTTPS_PROXY` and `NO_PROXY`.

### Logs

By default the agent logs each op it receives at info level: the command it runs, or the path it reads or writes, with the run it belongs to. It also logs exit codes, the pids of launched jobs, and when each job exits. The status checks it repeats every few seconds for running jobs (`poll`, `read_output`) appear only in debug logs.

### Debug logs

`horus-agent run --debug` (or `HORUS_AGENT_DEBUG=1`) logs everything the agent does:
- every TCP and TLS connection and HTTP request;
- each poll, with how long it waited and how many ops it got;
- each op with its arguments, its duration and its result;
- every command it runs, with a preview of the output;
- file transfers, job pids, cancels, and results delivered.

Environment variables appear by name only, and their values are hidden wherever a command or its output repeats them. Passwords in URLs, presigned URL signatures, bearer tokens and `token=`, `password=`, `secret=` or `api_key=` values are hidden too. Tokens are never logged. Commands and paths are logged as given, so they can name projects or samples: read a log before you share it. Under systemd, journald's `MaxRetentionSec` sets how long logs are kept.

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
- Without `run_as`, task commands run as the agent's Unix user and can read its config file. Set `run_as` on any machine others use.

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
