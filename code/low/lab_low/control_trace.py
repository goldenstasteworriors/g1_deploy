"""JSONL deployment diagnostics, without disk I/O on the motor/estop thread."""
from datetime import datetime
import json
from pathlib import Path
import queue
import sys
import threading
import time

import numpy as np


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class ControlTrace:
    def __init__(self, directory, metadata):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / ("control_trace_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + ".jsonl")
        self.queue = queue.SimpleQueue()
        self.error = None
        self.closed = False
        self.thread = threading.Thread(target=self._write, name="lab-control-trace", daemon=True)
        self.thread.start()
        self.emit(dict(event="schema", version=1, units=dict(q="rad", dq="rad/s", tau="Nm"), **metadata))
        print(f"[控制记录] {self.path}", flush=True)

    def emit(self, record):
        if not self.closed and self.error is None:
            # Freeze mutable arrays before the motor/state buffers are reused.
            try:
                self.queue.put(json.dumps(dict(wall_time_ns=time.time_ns(), monotonic_ns=time.monotonic_ns(),
                                               **record), default=json_value, allow_nan=False))
            except Exception as error:
                self._failed(error)

    def _failed(self, error):
        if self.error is None:
            self.error = str(error)
            print(f"[控制记录失败] {self.path}: {error}", file=sys.stderr, flush=True)

    def _write(self):
        try:
            with self.path.open("x", encoding="utf-8", buffering=1) as stream:
                while True:
                    line = self.queue.get()
                    if line is None:
                        break
                    stream.write(line + "\n")
        except Exception as error:
            self._failed(error)

    def close(self):
        if not self.closed:
            self.closed = True
            self.queue.put(None)
            self.thread.join()


class TracedPublisher:
    """Snapshot the actual LowCmd passed to DDS, including native damping."""
    def __init__(self, publisher, trace, context, motor_indices):
        self.publisher, self.trace, self.context = publisher, trace, context
        self.motor_indices = list(motor_indices)

    def __getattr__(self, name):
        return getattr(self.publisher, name)

    def Write(self, message):
        record = None
        try:
            record = dict(event="motor_command", **self.context())
            for field in ("q", "dq", "kp", "kd", "tau"):
                record["sent_arm_" + field] = [float(getattr(message.motor_cmd[i], field)) for i in self.motor_indices]
        except Exception as error:
            self.trace._failed(error)
        try:
            started = time.monotonic_ns()
            result = self.publisher.Write(message)
        except BaseException as error:
            if record is not None:
                self.trace.emit(dict(record, local_write_ok=False, write_error=str(error)))
            raise
        if record is not None:
            self.trace.emit(dict(record, send_started_monotonic_ns=started,
                                 send_finished_monotonic_ns=time.monotonic_ns(),
                                 local_write_ok=None if result is None else bool(result)))
        return result
