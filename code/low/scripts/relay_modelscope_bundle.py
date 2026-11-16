"""Stream ModelScope bytes over SSH without staging weights on the local disk.

Use when the target's Internet connection is slow. The remote sink writes only
inside --bundle and verifies every manifest hash. Credentials stay in memory.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import errno
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def sink(root, resume_sparse=False):
    root.mkdir(parents=True, exist_ok=True)
    root = root.resolve()
    stream = sys.stdin.buffer
    manifest = json.loads(stream.readline())
    completed, partial, partial_chunks = [], {}, {}
    previous = root / ".relay_manifest.json"
    if not previous.exists():
        previous = root / "MANIFEST.json"
    old_files = json.loads(previous.read_text()).get("files", {}) if previous.exists() else {}
    for name, meta in manifest["files"].items():
        target = (root / name).resolve()
        if not target.is_relative_to(root):
            raise ValueError("Bundle path escapes destination")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_file() and target.stat().st_size == meta["bytes"] and digest(target) == meta["sha256"]:
            completed.append(name)
        else:
            part = target.with_name(target.name + ".relay.part")
            if part.is_file() and part.stat().st_size == meta["bytes"] and digest(part) == meta["sha256"]:
                part.rename(target)
                completed.append(name)
                continue
            partial[name] = part
            # Recover only our sparse relay files bound to the SAME manifest.
            # This is a transport hint, never proof of correctness: the full
            # manifest SHA256 is still required before any file is committed.
            if resume_sparse and part.is_file() and old_files.get(name) == meta:
                ranges = []
                with part.open("rb") as f:
                    position, size = 0, part.stat().st_size
                    while position < size:
                        try:
                            begin = os.lseek(f.fileno(), position, os.SEEK_DATA)
                            end = os.lseek(f.fileno(), begin, os.SEEK_HOLE)
                        except OSError as error:
                            if error.errno == errno.ENXIO:
                                break
                            raise
                        ranges.append((begin, end))
                        position = end
                partial_chunks[name] = [offset for offset in range(0, meta["bytes"], 32 << 20)
                    if any(begin <= offset and end >= min(offset + (32 << 20), meta["bytes"])
                           for begin, end in ranges)]
    (root / ".relay_manifest.json").write_text(json.dumps(manifest))
    print(json.dumps({"completed": completed, "partial_chunks": partial_chunks}), flush=True)
    while True:
        header = json.loads(stream.readline())
        if header.get("done"):
            break
        name, offset, count = header["name"], header["offset"], header["bytes"]
        if name not in partial or offset < 0 or count < 1 or offset + count > manifest["files"][name]["bytes"]:
            raise ValueError("Invalid chunk")
        part = partial[name]
        with part.open("r+b" if part.exists() else "w+b") as f:
            f.seek(offset)
            remaining = count
            while remaining:
                block = stream.read(min(1 << 20, remaining))
                if not block:
                    raise EOFError("Truncated SSH stream")
                f.write(block)
                remaining -= len(block)
        print('{"ack":true}', flush=True)
    for name, part in partial.items():
        expected = manifest["files"][name]
        if part.stat().st_size != expected["bytes"] or digest(part) != expected["sha256"]:
            raise ValueError(f"Invalid completed file: {name}")
        part.rename(root / name)
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"verified_files": len(manifest["files"]), "download_source": "ModelScope via SSH relay"}), flush=True)


def relay(args):
    import requests
    from modelscope_hub import HubApi
    token = sys.stdin.read().strip()
    if not token:
        raise ValueError("Token required on stdin")
    api = HubApi(token=token)
    def url(name):
        return api.legacy.get_download_url(args.repo, "model", name, "master")
    session = requests.Session()
    session.trust_env = False
    session.cookies.set("m_session_id", token, domain="modelscope.cn")
    response = session.get(url("MANIFEST.json"), timeout=(30, 180))
    response.raise_for_status()
    manifest = response.json()
    command = ["ssh", "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=10"]
    if args.control_path:
        command += ["-o", "ControlPath=" + args.control_path]
    command += ["--", args.host, shlex.join([args.remote_python, args.remote_script, "--sink", "--bundle", str(args.bundle)])]
    if args.resume_sparse:
        command[-1] += " --resume-sparse"
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    process.stdin.write(json.dumps(manifest).encode() + b"\n")
    process.stdin.flush()
    inventory = json.loads(process.stdout.readline())
    completed = set(inventory["completed"])
    partial_chunks = inventory.get("partial_chunks", {})
    tasks = queue.Queue()
    chunk = 32 << 20
    for name, meta in manifest["files"].items():
        if name not in completed:
            for offset in range(0, meta["bytes"], chunk):
                if offset not in partial_chunks.get(name, []):
                    tasks.put((name, offset, min(chunk, meta["bytes"] - offset)))
    print(json.dumps({"already_verified_files": len(completed), "remaining_chunks": tasks.qsize(),
                      "reused_sparse_chunks": sum(map(len, partial_chunks.values()))}), flush=True)
    writer = threading.Lock()
    failed = threading.Event()
    transferred, last_log = [0], [time.monotonic()]
    def worker():
        client = requests.Session()
        client.trust_env = False
        client.cookies.set("m_session_id", token, domain="modelscope.cn")
        resolved_urls = {}
        while not failed.is_set():
            try:
                name, offset, count = tasks.get_nowait()
            except queue.Empty:
                return
            payload = bytearray()
            for attempt in range(12):
                try:
                    for delta in range(len(payload), count, args.request_mib << 20):
                        begin = offset + delta
                        length = min(args.request_mib << 20, count - delta)
                        with client.get(resolved_urls.get(name) or url(name), headers={"Range": f"bytes={begin}-{begin+length-1}"},
                                        stream=True, timeout=(30, 120)) as result:
                            if result.status_code not in (200, 206):
                                resolved_urls.pop(name, None)
                                raise RuntimeError("Download HTTP " + str(result.status_code))
                            resolved_urls[name] = result.url
                            if result.status_code == 200 and (begin != 0 or length != manifest["files"][name]["bytes"]):
                                raise ValueError("Server ignored range")
                            received = 0
                            segment = bytearray()
                            for block in result.iter_content(1 << 20):
                                received += len(block)
                                if received > length:
                                    raise ValueError("Oversized range response")
                                segment.extend(block)
                            if received != length:
                                raise ValueError("Truncated range response")
                            payload.extend(segment)
                    with writer:
                        process.stdin.write(json.dumps({"name": name, "offset": offset, "bytes": count}).encode() + b"\n")
                        process.stdin.write(payload)
                        process.stdin.flush()
                        if not json.loads(process.stdout.readline()).get("ack"):
                            raise RuntimeError("Missing SSH acknowledgement")
                        transferred[0] += count
                        if time.monotonic() - last_log[0] > 20:
                            print(json.dumps({"downloaded_GB": round(transferred[0] / 1e9, 3), "remaining_chunks": tasks.qsize()}), flush=True)
                            last_log[0] = time.monotonic()
                    break
                except (requests.RequestException, ValueError, RuntimeError) as error:
                    print(json.dumps({"retry": attempt + 1, "file": name, "offset": offset,
                                      "buffered_bytes": len(payload), "error_type": type(error).__name__}), flush=True)
                    if attempt == 11:
                        failed.set()
                        # Never log signed URLs, request objects or credentials.
                        raise RuntimeError(f"Download failed for {name}: {type(error).__name__}") from None
                    time.sleep(min(10, attempt + 1))
            tasks.task_done()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(worker) for _ in range(args.workers)]
            for future in futures:
                future.result()
        process.stdin.write(b'{"done":true}\n')
        process.stdin.flush()
        receipt = json.loads(process.stdout.readline())
        process.stdin.close()
        if process.wait() or receipt.get("verified_files") != len(manifest["files"]):
            raise RuntimeError("Remote verification failed")
        print(json.dumps(receipt), flush=True)
    finally:
        if process.poll() is None:
            process.terminate()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sink", action="store_true")
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--repo", default="ykjykj/LabWAM-BoxedV3-30k-20261111")
    p.add_argument("--host")
    p.add_argument("--control-path")
    p.add_argument("--remote-python", default="python3")
    p.add_argument("--remote-script")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--request-mib", type=int, choices=[1, 2, 4, 8, 16, 32], default=32,
                   help="HTTP range size within each atomic 32 MiB relay chunk")
    p.add_argument("--resume-sparse", action="store_true", help="Reuse occupied chunks in this relay's same-manifest sparse files; always verify complete SHA256")
    args = p.parse_args()
    if args.sink:
        sink(args.bundle, args.resume_sparse)
    else:
        if not args.host or not args.remote_script:
            p.error("--host and --remote-script required")
        relay(args)


if __name__ == "__main__":
    main()
