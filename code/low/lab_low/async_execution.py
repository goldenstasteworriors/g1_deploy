"""Depth-one action prefetch adapted from OpenWAM AsyncInferenceExecutor.

Same remaining-buffer trigger and returned-action-step trimming. poll() never blocks
the command pump; an unfinished future means hold the previous target. No
motor transport or model execution lives here.
"""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import math

import numpy as np


class AsyncChunkExecutor:
    def __init__(self, generate, *, horizon=32, delay_steps=16, simulate_latency=False,
                 smoothing_steps=0, smoothing_mask=None, initial_target=None,
                 freeze_simulation_on_inference=False):
        if not 1 <= horizon <= 32 or not 0 <= delay_steps < horizon:
            raise ValueError("Require 1 <= horizon <= 32 and 0 <= delay_steps < horizon")
        self.generate, self.horizon, self.delay_steps = generate, horizon, delay_steps
        self.simulate_latency = simulate_latency
        if freeze_simulation_on_inference and not simulate_latency:
            raise ValueError("Frozen inference clock is only valid for offline simulation")
        self.freeze_simulation_on_inference = freeze_simulation_on_inference
        if type(smoothing_steps) is not int or not 0 <= smoothing_steps <= 32:
            raise ValueError("smoothing_steps must be an integer in 0..32")
        self.smoothing_steps = smoothing_steps
        self.smoothing_mask = np.ones(16,dtype=bool) if smoothing_mask is None else np.asarray(smoothing_mask,dtype=bool).copy()
        if self.smoothing_mask.shape != (16,):
            raise ValueError("Expected 16D smoothing mask")
        # Dex1 targets retain their exact model values and timing.
        self.smoothing_mask[[7,15]] = False
        self.last_target = None if initial_target is None else np.asarray(initial_target,dtype=float).copy()
        if self.last_target is not None and (self.last_target.shape != (16,) or not np.isfinite(self.last_target).all()):
            raise ValueError("Expected finite 16D initial target")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="labwam-prefetch")
        self.buffer = deque()
        self.future = None
        self.pending_step = None
        self.pending_clock_step = None
        self.closed = False
        self.started = False
        self.adoptions = []
        self.warmup_ticks = self.underrun_ticks = 0

    def _start(self, step, snapshot, clock_step):
        condition = snapshot(step)
        self.pending_step = step
        self.pending_clock_step = clock_step
        self.future = self.pool.submit(self.generate, condition)

    def poll(self, step, snapshot, *, clock_step=None):
        # Like official _current_step, `step` counts returned model actions.
        # The separate clock counts 30Hz hold ticks for latency and history.
        clock_step = step if clock_step is None else clock_step
        if self.closed:
            return None
        if not self.buffer:
            if self.future is None:
                self._start(step, snapshot, clock_step)
            ready = self.future.done()
            # Offline rendering excludes the SSH proxy's transport overhead:
            # wait in wall time, then account for measured pipeline latency on
            # the simulated clock. The real robot never takes this path.
            result = self.future.result() if ready or self.simulate_latency else None
            if self.simulate_latency:
                ready = True
            # Rendering may run slower than real time. Do not make simulated
            # model latency look shorter just because rendering is expensive.
            if ready and self.simulate_latency and not self.freeze_simulation_on_inference:
                elapsed = clock_step-self.pending_clock_step if self.started else self.warmup_ticks
                ready = elapsed >= math.ceil(result['pipeline_seconds']*30)
            if not ready:
                if self.started:
                    self.underrun_ticks += 1
                else:
                    self.warmup_ticks += 1
                return None
            actions = np.asarray(result['actions'], dtype=float)
            if actions.shape != (32,16) or not np.isfinite(actions).all():
                raise ValueError("Expected finite 32x16 action chunk")
            skip = max(0, step-self.pending_step) if self.started else 0
            end = min(len(actions),skip+self.horizon)
            if skip >= end:
                raise RuntimeError(f"Async inference result is stale: skip_steps={skip}, action_horizon={len(actions)}")
            metadata = dict(prediction_id=result['prediction_id'], input_step=self.pending_step,
                            skip_steps=skip, inference_mode="async")
            self.adoptions.append(dict(metadata, adoption_step=step, available=end-skip,
                                       adoption_clock_step=clock_step,
                                       pipeline_seconds=result['pipeline_seconds']))
            offset = np.zeros(16)
            if self.smoothing_steps and self.last_target is not None:
                offset[self.smoothing_mask] = (self.last_target-actions[skip])[self.smoothing_mask]
            for j,i in enumerate(range(skip,end)):
                raw = actions[i]
                # Quintic decay of the boundary offset. First target equals the
                # previously sent arm pose; the raw trajectory is recovered at
                # the end of the window. No velocity clipping or watchdog.
                u = min(1.,j/max(1,self.smoothing_steps-1))
                weight = 1-(10*u**3-15*u**4+6*u**5) if self.smoothing_steps else 0.
                smoothed = raw+weight*offset
                self.buffer.append((smoothed,dict(metadata,action_index=i,
                    model_target16=raw.tolist(),smoothed_target16=smoothed.tolist(),
                    smoothing_weight=weight,smoothing_steps=self.smoothing_steps)))
            self.future, self.pending_step, self.pending_clock_step = None, None, None
            self.started = True
        if self.future is None and len(self.buffer) <= self.delay_steps:
            self._start(step,snapshot,clock_step)
        item = self.buffer.popleft()
        self.last_target = item[0].copy()
        return item

    def close(self):
        self.closed = True
        self.buffer.clear()
        if self.future is not None:
            self.future.cancel()
        # A running RPC only produces a value. It never publishes an action.
        self.pool.shutdown(wait=False,cancel_futures=True)

    def stats(self):
        return dict(warmup_hold_ticks=self.warmup_ticks,underrun_hold_ticks=self.underrun_ticks,
                    chunks_adopted=len(self.adoptions),adoptions=list(self.adoptions),
                    smoothing_steps=self.smoothing_steps,smoothing_mask=self.smoothing_mask.tolist())
