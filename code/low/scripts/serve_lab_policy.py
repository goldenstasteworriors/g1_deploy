"""HTTP transport for the serialized LabWAM worker; no robot commands."""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab_low.inference_optimizations import (add_optimization_arguments, options_from_arguments,
                                           worker_optimization_arguments)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8849)
    add_optimization_arguments(p)
    args = p.parse_args()
    options_from_arguments(args)  # validate before loading the model
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    worker = subprocess.Popen([sys.executable, "-u", str(Path(__file__).with_name("deploy_lab_worker.py")),
                               "--checkpoint", args.checkpoint, "--steps", str(args.steps),
                               *worker_optimization_arguments(args)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
    try:
        serve(worker, args)
    except KeyboardInterrupt:
        pass
    finally:
        # Also runs if interrupted while loading, before the HTTP server exists.
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)
        worker.stdin.close()
        worker.stdout.close()


def serve(worker, args):
    ready = json.loads(worker.stdout.readline())
    if not ready.get("ready"):
        raise RuntimeError("Policy failed to load")
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, payload):
            body = json.dumps(payload, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            returncode = worker.poll()
            alive = returncode is None
            self.respond(200 if alive else 503,
                         {**ready, "ready": alive and bool(ready.get("ready")),
                          "alive": alive, "worker_returncode": returncode, "execution_allowed": False})

        def do_POST(self):
            if self.path != "/rpc":
                self.respond(404, {"ok": False})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0:
                    raise ValueError("Request body must be nonempty")
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                with lock:
                    if worker.poll() is not None:
                        raise RuntimeError("Worker stopped")
                    worker.stdin.write(json.dumps(payload, allow_nan=False) + "\n")
                    worker.stdin.flush()
                    result = json.loads(worker.stdout.readline())
                if payload.get("op") in ("reset", "stop") or not result.get("ok"):
                    print(json.dumps(dict(event="policy_rpc", op=payload.get("op"),
                        binding_id=payload.get("binding_id"), client_scope=payload.get("client_scope"),
                        ok=result.get("ok"), error=result.get("error"))), flush=True)
                self.respond(200 if result.get("ok") else 400, result)
            except (ValueError, RuntimeError, OSError) as error:
                self.respond(400, {"ok": False, "error": str(error)})

    server = ThreadingHTTPServer((args.bind, args.port), Handler)
    print(json.dumps({"listening": args.bind, "port": args.port, **ready}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
