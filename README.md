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

The same file holds the rest of the machine's policy; see [Policy and approved releases](#policy-and-approved-releases).

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
- A strict agent (the default) needs `[jobs] run_as` in its policy, so tasks cannot change the agent's own files. `sudo horus-agent policy init` writes a starting policy; `sudo horus-agent policy init --open` writes one for a machine that accepts any workflow.

## Policy and approved releases

The machine's administrator decides what the agent may do, in a root-owned policy file. The cloud cannot change it.

With no policy file the agent is **strict**: it runs only workflow releases approved on this machine, and nothing else. On a machine where any workflow may run, write `mode = "open"`. In strict mode tasks must run as their own user (`[jobs] run_as`); `allow_same_user = true` overrides that. `sudo horus-agent policy init [--open] [--run-as USER]` writes a starting file. The agent reads the policy when it starts: restart it after a change.

```toml
# /etc/horus-agent/policy.toml (owned by root, mode 0644)
schema = 1
mode = "strict"                 # strict | open
work_dir = "/srv/horus-agent"   # the agent's --work-dir must be this or under it
allowed_hosts = ["api.example.com", "bucket.s3.eu-north-1.amazonaws.com"]   # broker and storage bucket
uploads = false                 # only run metadata leaves the machine
logs = "off"                    # task output stays on the machine
downloads = true
resource_usage = true           # CPU and RAM samples for the run's charts
max_put_bytes = 4194304
max_fetch_bytes = 53687091200
max_concurrent_runs = 4
audit_log = "/var/log/horus-agent/ops.jsonl"   # default: <work_dir>/.horus-agent/ops.jsonl

[jobs]
run_as = "horus-job"            # see Run tasks as another user
run_as_via = "sudo"

[data]
read_roots = ["/data/sequencing"]   # extra dirs a path parameter may name
```

| Key | Default |
|---|---|
| `mode` | `strict` |
| `require_release` | `true` |
| `work_dir` | the registered work dir |
| `allowed_hosts` | the broker URL's host |
| `uploads` / `logs` | `false` / `off` |
| `downloads` / `resource_usage` | `true` / `true` |

The agent refuses to start if the policy file, `/etc/horus-agent/approved/` or a parent directory is not owned by root or can be written by group or others.

In both modes:

- every path an op names must sit in the work dir; in strict mode also in its own run's directory (the folder named after its run id, which must be a UUID), with symlinks resolved; the folders of two runs never overlap;
- downloads and uploads go over https to `allowed_hosts` only, with size caps and timeouts;
- every op decision is appended to the audit log (method, run, paths, allowed or refused; never data or environment values).

### Approve a release

```sh
sudo horus-agent approve <release-id>          # downloaded with the agent's token
sudo horus-agent approve ./release.zip         # a release archive exported from the UI
sudo horus-agent approve <release-id> --expect-hash <hash> --yes
horus-agent approvals                          # hash, name, version, when, by whom
sudo horus-agent revoke <hash>                 # refused within 10 s; running jobs killed within 30 s
horus-agent policy check                       # the effective policy, or what is wrong
```

`approve` reads the release on this machine and prints what it can run before asking for `yes`: each agent task's command, script hash, Slurm options, inputs (constant values and file hashes, run parameters) and secret names. The approval is stored under its content hash, so publishing the same content again needs no new approval, and any change to a command, script, constant file, Slurm option or secret name needs one. `approve` refuses a release whose commands use a placeholder anywhere but as a bare word (inside quotes, a heredoc, `${...}`, arithmetic, or the arguments of `eval`, `sh -c`, `ssh` and similar), or that holds control or invisible formatting characters, and every field it prints is escaped. Write `${x}`, never `"${x}"`: the agent quotes every value. `approve <release-id>` reads the agent's config from `HORUS_AGENT_CONFIG`, your own, or `/var/lib/horus-agent/.config/horus-agent/agent.json`.

### What a strict agent runs

- Tasks of approved releases with the `shell` executor and a `command` or `python_script` runtime, on `kind: agent` or `kind: slurm` with `inner: {kind: agent}`.
- The agent renders each task's command itself from the approved template. The orchestrator only says which approved task runs and binds its placeholders: a path in the run's directory, made of letters, digits and `._+,=@%:/-[]` only, or a value (a run parameter, or a value from an upstream task), which the agent renders as one quoted word.
- Constant inputs and scripts are copied into the run's `.horus-const` folder as they are hashed against the approval, and the command names those copies, which no op can reach. Constant folders may hold only files and folders. A task runs in its own working directory inside its run's folder, as the runtime runs it, so relative outputs land where the runtime collects them; that directory must be one the agent made and not reached through a link. Files placed there cannot replace what runs: scripts and constants run from the agent's copies, and a `python_script` runs with `python -I`, so nothing in the working directory is imported.
- Secrets reach the task through its shell's stdin, never a file or its command line, and only under the names the release declares.
- For Slurm the agent writes the batch script from the approved Slurm options and runs `sbatch`, `squeue`, `sacct` and `scancel` itself.
- File plumbing (hashing, packing and unpacking folders, moves, sizes) has native ops in the agent (`digest`, `pack`, `unpack`, `move`, `stat`) instead of shell commands; `AgentTarget` implements the runtime's file hooks with them: `path_digest`, `pack_folder`, `unpack_folder`, `move_file` and `file_size`. Archives are unpacked without absolute paths, `..`, links out or device files.
- Job pids and Slurm job ids come from the agent's own records, never from files in the run's directory, and each job gets a job dir the agent creates.
- With `uploads = false`, only metadata the agent or the orchestrator wrote (pid and exit code of a job, resource samples, the runtime's `.horus/*.json`) can be read back; no op may write data under those names. With `logs = "off"`, job logs stay on the machine whatever `uploads` says.
- After `revoke`, nothing new of the release starts, and within about 30 s the agent kills the running jobs (and cancels the Slurm jobs) of runs bound to it. Those runs can still be polled and cleaned up.
- No raw commands: `exec` and `launch` are refused, and file reads and writes never follow a symlink in the file's own name. A refused op fails the task with `PolicyRefusedError`, an `OperationRefusedError` of the runtime. Shared Slurm allocations, containers and Python-function executors are not accepted in approved releases.
- The agent samples each task's CPU and RAM and writes the files the resource monitor reads, so runs keep their charts.

The orchestrator needs horus-agent 0.2 or newer to talk to a strict agent, and a horus-runtime that calls the target's native file hooks instead of shell commands for staging and hashing.

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
- A strict agent runs only approved releases (see above).
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
