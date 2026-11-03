#
# horus_agent
# Copyright (c) 2026 Temple Compute
#
# Commercial use and distribution of this software is prohibited.
# For commercial use, please contact the author for licensing options.
#
"""
In-memory broker implementing the protocol in ``horus_agent.protocol``,
served over real HTTP so the target and the agent talk to it exactly as
they would to a production broker.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

RUN_TOKEN = "run-token"
REGISTRATION_TOKEN = "register-me"
RUN_ID = "run-1"


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request: Any, client_address: Any) -> None:
        """Ignore clients hanging up mid-reply (a stopping agent's poll)."""
        del request, client_address


class FakeBroker:
    """Broker state plus the HTTP server in front of it."""

    def __init__(self, max_wait: float = 0.5, host: str = "127.0.0.1") -> None:
        self.max_wait = max_wait
        self.cond = threading.Condition()
        self.agent_tokens: dict[str, str] = {}
        self.queues: dict[str, list[dict[str, Any]]] = {}
        self.op_agent: dict[str, str] = {}
        self.delivered: dict[str, str] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, str] = {}
        self.seen: dict[str, float] = {}
        self.work_dirs: dict[str, str | None] = {}
        self.blobs: dict[str, bytes] = {}
        self.ops_seen: list[dict[str, Any]] = []
        self.fail_results = 0
        self.fail_polls = 0
        self.max_results_per_post = 0
        """When set, a larger results batch is refused with 413."""
        self.server = _Server((host, 0), _handler(self))
        self.public_url: str | None = None
        """Where the agent reaches this broker, when not :attr:`url`."""
        self.thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    @property
    def url(self) -> str:
        host = str(self.server.server_address[0])
        return (
            f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{self.port}"
        )

    def start(self) -> "FakeBroker":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def add_agent(self, name: str, token: str) -> None:
        with self.cond:
            self.agent_tokens[token] = name
            self.queues.setdefault(name, [])

    def enqueue(self, agent: str, op: dict[str, Any]) -> None:
        """Queue *op* for *agent* the way the broker does (sets run_id)."""
        with self.cond:
            self.ops_seen.append(op)
            self.op_agent[op["op_id"]] = agent
            self.queues.setdefault(agent, []).append({**op, "run_id": RUN_ID})
            self.cond.notify_all()

    def status(self, op_id: str) -> dict[str, Any]:
        agent = self.op_agent[op_id]
        if op_id in self.results:
            result = self.results[op_id]
            return {
                "done": True,
                "result": result.get("result"),
                "error": result.get("error"),
            }
        seen = self.seen.get(agent)
        delivered = self.delivered.get(op_id)
        return {
            "done": False,
            "agent_seen_ago": None if seen is None else time.time() - seen,
            "lost": delivered is not None
            and delivered != self.sessions.get(agent),
        }

    def wait_status(self, op_id: str, wait: float) -> dict[str, Any]:
        deadline = time.time() + min(wait, self.max_wait)
        with self.cond:
            while op_id not in self.results and time.time() < deadline:
                self.cond.wait(deadline - time.time())
            return self.status(op_id)


def _handler(broker: FakeBroker) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args

        def _json_body(self) -> Any:
            length = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(length) or b"null")

        def _reply(self, code: int, body: Any = None) -> None:
            data = b"" if body is None else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _bearer(self) -> str:
            return self.headers.get("Authorization", "").removeprefix(
                "Bearer "
            )

        def do_PUT(self) -> None:
            key = urlparse(self.path).path.removeprefix("/blobs/")
            length = int(self.headers["Content-Length"])
            broker.blobs[key] = self.rfile.read(length)
            self._reply(200)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            if url.path.startswith("/blobs/"):
                data = broker.blobs.get(url.path.removeprefix("/blobs/"))
                if data is None:
                    self._reply(404)
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
                return
            if self._bearer() != RUN_TOKEN:
                self._reply(401, {"detail": "bad token"})
                return
            if url.path.startswith("/agents/info/"):
                name = url.path.removeprefix("/agents/info/")
                if name not in broker.queues:
                    self._reply(404, {"detail": f"No agent named {name!r}"})
                    return
                self._reply(
                    200,
                    {
                        "name": name,
                        "work_dir": broker.work_dirs.get(name),
                        "online": name in broker.seen,
                    },
                )
                return
            op_id = url.path.removeprefix("/agents/ops/")
            wait = float(parse_qs(url.query).get("wait", ["0"])[0])
            self._reply(200, broker.wait_status(op_id, wait))

        def do_POST(self) -> None:  # noqa: PLR0911
            path = urlparse(self.path).path
            body = self._json_body()
            if path == "/agents/register":
                if body["token"] != REGISTRATION_TOKEN:
                    self._reply(401, {"detail": "Invalid or used token"})
                    return
                token = f"tca_{uuid4().hex}"
                name = body.get("name") or body["hostname"]
                broker.add_agent(name, token)
                broker.work_dirs[name] = body.get("work_dir")
                self._reply(
                    201,
                    {
                        "agent_id": uuid4().hex,
                        "name": name,
                        "agent_token": token,
                    },
                )
                return
            if path.startswith("/agents/me/"):
                agent = broker.agent_tokens.get(self._bearer())
                if agent is None:
                    self._reply(401, {"detail": "bad agent token"})
                    return
                if path == "/agents/me/poll":
                    self._poll(agent, body)
                else:
                    self._results(body)
                return
            if self._bearer() != RUN_TOKEN:
                self._reply(401, {"detail": "bad token"})
                return
            if path == "/agents/staging/":
                key = uuid4().hex
                url = f"{broker.url}/blobs/{key}"
                public = f"{broker.public_url or broker.url}/blobs/{key}"
                self._reply(
                    200,
                    {
                        "orchestrator_put_url": url,
                        "orchestrator_get_url": url,
                        "agent_put_url": public,
                        "agent_get_url": public,
                    },
                )
                return
            agent, op = body["agent"], body["op"]
            if agent not in broker.queues:
                self._reply(404, {"detail": f"No agent named {agent!r}"})
                return
            if op["op_id"] not in broker.op_agent:
                broker.enqueue(agent, op)
            self._reply(200, broker.wait_status(op["op_id"], 20))

        def _poll(self, agent: str, body: dict[str, Any]) -> None:
            with broker.cond:
                if broker.fail_polls:
                    broker.fail_polls -= 1
                    self._reply(500)
                    return
                broker.sessions[agent] = body["session"]
                broker.seen[agent] = time.time()
                if body.get("work_dir"):
                    broker.work_dirs[agent] = body["work_dir"]
                deadline = time.time() + broker.max_wait
                queue = broker.queues[agent]
                while not queue and time.time() < deadline:
                    broker.cond.wait(deadline - time.time())
                ops, queue[:] = list(queue), []
                for op in ops:
                    broker.delivered[op["op_id"]] = body["session"]
            if ops:
                self._reply(200, {"ops": ops})
            else:
                self._reply(204)

        def _results(self, body: dict[str, Any]) -> None:
            with broker.cond:
                if broker.fail_results:
                    broker.fail_results -= 1
                    self._reply(500)
                    return
                limit = broker.max_results_per_post
                if limit and len(body["results"]) > limit:
                    self._reply(413)
                    return
                for result in body["results"]:
                    broker.results[result["op_id"]] = result
                broker.cond.notify_all()
            self._reply(204)

    return Handler
