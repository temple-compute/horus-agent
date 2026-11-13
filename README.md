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
| `--ca-bundle PEM` | `ca_bundle` | Trust only these CAs for the broker. Default: `SSL_CERT_FILE` or `SSL_CERT_DIR` when set, else the certifi bundle |
| `--job-env NAME` | `job_env` | Pass this variable on to tasks (repeatable) |

To change a setting later, edit `agent.json` and restart the agent, for example:

```sh
python3 -c 'import json,sys; p=sys.argv[1]; c=json.load(open(p)); c["job_env"]=["MODULEPATH","CONDA_EXE"]; json.dump(c,open(p,"w"))' ~/.config/horus-agent/agent.json
```

Which user runs the tasks is set in a root-owned file instead (see [Run tasks as another user](#run-tasks-as-another-user)).

### Pinned install

Each GitHub release has a `requirements.txt` that pins horus-agent and every dependency by hash:

```sh
python3.13 -m venv /opt/horus-agent      # or: conda create -p /opt/horus-agent python=3.13 pip
/opt/horus-agent/bin/pip install --require-hashes -r requirements.txt
```

### Defaults

- The agent refuses to run as root; `--allow-root` overrides it, and tasks then run as root too.
- Files the agent and its tasks create are private to the agent's user (umask 077), and the work dir loses any access for other users.
- Tasks get a minimal environment: `PATH`, `HOME`, `USER`, `LOGNAME`, `SHELL`, `LANG`, `LANGUAGE`, `LC_*`, `TZ`, `TERM`, `TMPDIR`, `SSL_CERT_FILE`, `SSL_CERT_DIR` and the proxy variables, plus `job_env` and the variables the workflow sets. At start the agent logs the names of the variables tasks do not get.
- Variables the workflow sets reach the task through its shell's stdin, never a command line, so other users cannot see them in `ps`.
- `poll` and `signal` act only on jobs the agent launched, and the agent never follows a symlink when it reads a job's output, exit code or pid.

### Run tasks as another user

Without `run_as`, task commands run as the agent's user and can read its token. On a shared or production machine, give tasks their own user. The agent and the task user share the work dir through the task user's group:

```sh
useradd --system --home-dir /var/lib/horus-agent --shell /usr/sbin/nologin horus-agent
useradd --system --home-dir /srv/horus-agent --no-create-home --shell /usr/sbin/nologin horus-job
usermod -aG horus-job horus-agent
install -d -o horus-agent -g horus-agent -m 700 /var/lib/horus-agent
install -d -o horus-agent -g horus-job -m 2770 /srv/horus-agent
```

Then name the task user in `/etc/horus-agent/policy.toml`. The agent refuses to start unless this file and every directory above it are owned by root and not writable by group or others, so the agent's user cannot change it:

```sh
install -d -m 755 /etc/horus-agent
printf '[jobs]\nrun_as = "horus-job"\nrun_as_via = "sudo"\n' > /etc/horus-agent/policy.toml
chmod 644 /etc/horus-agent/policy.toml
```

With `run_as` set, files are shared with that group (umask 007), never with other users. The token stays in `/var/lib/horus-agent`, which the task user cannot open. Tasks get the task user's `HOME` (here `/srv/horus-agent`, so caches such as `~/.cache` and `~/.conda` land in the work dir).

The agent switches user with `/usr/bin/sudo` or `/usr/bin/setpriv`, by absolute path and with a fixed environment, and the task's environment is applied only after the switch:

- `sudo` (default, recommended): the agent itself holds no privilege. Add this rule as `/etc/sudoers.d/horus-agent`:

  ```
  horus-agent ALL=(horus-job) NOPASSWD: ALL
  ```

- `setpriv` (util-linux): the agent holds `CAP_SETUID` and `CAP_SETGID` and drops them for the task. Anyone who takes over the agent's process can then become any user, so prefer `sudo`.

Do not add the agent or task user to the `docker` group: that is root on the machine. Use rootless Podman or Apptainer for container tasks.

### Run it as a system service

`examples/horus-agent.service` is a hardened system unit: `NoNewPrivileges`, `ProtectSystem=strict` with only `/srv/horus-agent` writable, a private `/tmp`, read-only home directories, no capabilities, and Python started with `-I` so nothing in a writable directory is imported. Tasks inherit all of it. With the users and policy above:

```sh
python3.13 -m venv /opt/horus-agent
/opt/horus-agent/bin/pip install --require-hashes -r requirements.txt
sudo -u horus-agent env HORUS_AGENT_CONFIG=/var/lib/horus-agent/agent.json \
    /opt/horus-agent/bin/horus-agent register --url https://api.example.com/api/v1 \
    --work-dir /srv/horus-agent
cp examples/horus-agent.service /etc/systemd/system/
mkdir -p /etc/systemd/system/horus-agent.service.d
cp examples/horus-agent.service.d/run-as-sudo.conf /etc/systemd/system/horus-agent.service.d/
systemctl enable --now horus-agent
```

Drop-ins in `examples/horus-agent.service.d/`:

| File | When |
|---|---|
| `run-as-sudo.conf` | `run_as_via = "sudo"`: lets `sudo` start (no `NoNewPrivileges`, only the capabilities sudo needs) |
| `run-as-setpriv.conf` | `run_as_via = "setpriv"` |
| `podman.conf` | Rootless Podman tasks, with one of the two above: `newuidmap` and a container's `/proc` need what these settings turn off |

The work dir must be under `ReadWritePaths` (the agent refuses to start otherwise). To keep an existing work dir, such as `~/horus-agent`, add it there with a drop-in. Without `run_as`, create `/srv/horus-agent` with `-g horus-agent -m 700`, skip the `horus-job` user and the drop-ins; tasks then use `/var/lib/horus-agent` as `HOME`.

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

### Upgrading from 0.1

- The agent no longer runs as root. Move it to its own user (above), or pass `--allow-root` to `run` for now.
- The broker URL must use `https://` (or `http://localhost`). Change `url` in `agent.json` if it does not.
- A broker behind a TLS-inspecting proxy keeps working with `SSL_CERT_FILE`; to pin that CA for the broker only, set `ca_bundle` in `agent.json`. A failed certificate check names `ca_bundle` in the log.
- Tasks no longer get the agent's whole environment. The first `run` logs the names they lose; add the ones they need (for example `MODULEPATH`, `LD_LIBRARY_PATH`, `CONDA_EXE`, `XDG_RUNTIME_DIR`) to `job_env`.
- `agent.json` must be mode 0600 (`chmod 600 ~/.config/horus-agent/agent.json`).
- Under the hardened system unit, the work dir must be under `ReadWritePaths`.

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
