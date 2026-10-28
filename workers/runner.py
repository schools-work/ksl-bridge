"""Redis Streams consumer: micro-batching, deadlines (drop stale jobs), XAUTOCLAIM recovery, replies."""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import time
from typing import Any

import msgpack
import numpy as np
import redis.asyncio as aioredis

from .adapters import ModelAdapter, ModelUnavailable
from .health import heartbeat_forever


def _np_default(o: Any) -> Any:
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


class StreamWorker:
    def __init__(
        self,
        redis: aioredis.Redis,
        task: str,
        adapter: ModelAdapter,
        batch_max: int = 32,
        batch_wait_ms: int = 15,
        consumer: str | None = None,
    ):
        self.r, self.task, self.adapter = redis, task, adapter
        self.stream, self.group = f"jobs:{task}", f"workers:{task}"
        self.consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self.batch_max, self.wait = batch_max, batch_wait_ms / 1000

    async def ensure_group(self) -> None:
        try:
            await self.r.xgroup_create(self.stream, self.group, id="$", mkstream=True)
        except aioredis.ResponseError as e:
            if "BUSYGROUP" not in str(e):
                raise

    async def _reply(self, job: dict[str, Any], output: Any, ok: bool = True, error: str | None = None) -> None:
        await self.r.publish(
            job["reply_to"],
            msgpack.packb(
                {"job_id": job["job_id"], "ok": ok, "output": output, "error": error},
                default=_np_default,
                use_bin_type=True,
            ),
        )

    async def _janitor(self) -> None:
        """Reclaim jobs idle > 5 s from dead consumers; stale ones are acked and dropped."""
        while True:
            await asyncio.sleep(5)
            try:
                _, claimed, *_ = await self.r.xautoclaim(
                    self.stream, self.group, self.consumer, min_idle_time=5000, start_id="0-0", count=100
                )
                for msg_id, fields in claimed:
                    job = msgpack.unpackb(fields[b"p"], raw=False)
                    if job.get("deadline_ms", 0) < time.time() * 1000:
                        await self.r.xack(self.stream, self.group, msg_id)
            except aioredis.ResponseError:
                pass

    async def run(self, stop: asyncio.Event | None = None) -> None:
        await self.ensure_group()
        await asyncio.to_thread(self.adapter.load)
        await asyncio.to_thread(self.adapter.warmup)
        hb = asyncio.create_task(heartbeat_forever(self.r, self.task))
        jan = asyncio.create_task(self._janitor())
        try:
            while stop is None or not stop.is_set():
                batch = await self._collect()
                if batch:
                    await self._process(batch)
        finally:
            hb.cancel()
            jan.cancel()

    async def _collect(self) -> list[tuple[bytes, dict[str, Any]]]:
        batch: list[tuple[bytes, dict[str, Any]]] = []
        deadline = time.monotonic() + self.wait
        first = True
        while len(batch) < self.batch_max:
            block = 1000 if first else max(1, int((deadline - time.monotonic()) * 1000))
            resp = await self.r.xreadgroup(
                self.group, self.consumer, {self.stream: ">"}, count=self.batch_max - len(batch), block=block
            )
            for _, entries in resp or []:
                for msg_id, fields in entries:
                    batch.append((msg_id, msgpack.unpackb(fields[b"p"], raw=False)))
            if first and batch:
                deadline = time.monotonic() + self.wait
            first = False
            if not batch or time.monotonic() >= deadline:
                break
        return batch

    async def _process(self, batch: list[tuple[bytes, dict[str, Any]]]) -> None:
        now = time.time() * 1000
        live = [(i, j) for i, j in batch if j.get("deadline_ms", now + 1) > now]
        try:
            if live:
                outs = await asyncio.to_thread(self.adapter.infer_batch, [j["input"] for _, j in live])
                for (_, job), out in zip(live, outs, strict=True):
                    await self._reply(job, out)
        except ModelUnavailable as e:
            for _, job in live:
                await self._reply(job, None, ok=False, error=str(e))
        except Exception as e:  # one bad batch must not kill the worker
            for _, job in live:
                await self._reply(job, None, ok=False, error=type(e).__name__)
        finally:
            await self.r.xack(self.stream, self.group, *[i for i, _ in batch])


def main(task: str, adapter: ModelAdapter) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-max", type=int, default=int(os.environ.get("KSL_WORKER_BATCH_MAX", 32)))
    ap.add_argument("--batch-wait-ms", type=int, default=int(os.environ.get("KSL_WORKER_BATCH_WAIT_MS", 15)))
    args = ap.parse_args()
    r = aioredis.from_url(os.environ.get("KSL_REDIS_URL", "redis://localhost:6379/0"))
    asyncio.run(StreamWorker(r, task, adapter, args.batch_max, args.batch_wait_ms).run())
