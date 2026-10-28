"""/ws/v1: hello-first auth, room join, routing to the pipeline, heartbeat watchdog, resume."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.modules.conversation.application import join_room
from app.modules.feedback.application import record_feedback
from app.modules.identity.infrastructure.jwt import get_jwt
from app.modules.pipeline.application.session import PipelineSession
from app.platform.db import transaction
from app.platform.logging import get_logger
from app.platform.metrics import WS_CONNECTIONS, WS_MESSAGES
from app.platform.principal import Principal
from app.platform.ratelimit import POLICIES, RateLimiter
from app.settings import get_settings
from ksl_core._generated.ws_messages import ErrorCode
from ksl_core.errors import InvalidFrame
from ksl_core.landmarks import unpack

from . import presence, resume
from .connection import ClientConn
from .protocol import BIN_AUDIO, BIN_LANDMARKS, ProtocolError, error_msg, parse_client, server_msg

router = APIRouter()
log = get_logger(__name__)


async def _broadcast_state(ws: WebSocket, room: str) -> None:
    async with transaction(Principal(roles=frozenset({"admin"}))) as conn:
        state = await join_room.room_state(conn, UUID(room))
    if state:
        await ws.app.state.bus.publish(room, server_msg("room.state", {"members": state["members"]}, room=room))


@router.websocket("/ws/v1")
async def ws_endpoint(ws: WebSocket) -> None:
    s = get_settings()
    app = ws.app
    await ws.accept()
    try:
        env, hello = parse_client(await asyncio.wait_for(ws.receive_text(), timeout=5))
        if hello.type != "hello":
            raise ProtocolError(ErrorCode.PROTOCOL_VIOLATION, "hello must be first")
        principal: Principal = get_jwt().decode(hello.token)
    except Exception:
        await ws.close(code=int(ErrorCode.AUTH_REQUIRED))
        return

    resumed_id = (hello.resume or {}).get("session_id")
    session_id = resumed_id or uuid.uuid4().hex
    if resumed_id:
        own = await resume.owner(app.state.redis, resumed_id)
        if own is None or own[0] != principal.subject:
            session_id = uuid.uuid4().hex
            resumed_id = None
    conn = ClientConn(ws, session_id=session_id, subject=principal.subject)
    conn.last_peer_seq = env.seq
    writer = asyncio.create_task(conn.writer())
    WS_CONNECTIONS.inc()
    session = PipelineSession(
        session_id=session_id, principal=principal, broadcaster=app.state.bus, jobs=app.state.jobs, reply=conn.enqueue
    )
    conn.enqueue(
        server_msg(
            "welcome",
            {
                "session_id": session_id,
                "heartbeat_s": s.ws_heartbeat_seconds,
                "server_time": int(time.time() * 1000),
                "limits": {"max_frame_bytes": s.ws_max_frame_bytes},
            },
        )
    )
    if resumed_id:
        for text in await resume.load_after(app.state.redis, resumed_id, int(hello.resume.get("last_seq", 0))):
            await ws.send_text(text)
    limiter = RateLimiter(app.state.redis)
    room: str | None = None
    last_rx = time.monotonic()

    async def watchdog() -> None:
        while True:
            await asyncio.sleep(5)
            if time.monotonic() - last_rx > s.ws_idle_timeout_seconds:
                await ws.close(code=1001)
                return

    wd = asyncio.create_task(watchdog())

    try:
        while True:
            m = await ws.receive()
            if m.get("type") == "websocket.disconnect":
                break
            last_rx = time.monotonic()
            if m.get("bytes") is not None:
                b: bytes = m["bytes"]
                if room is None or len(b) > s.ws_max_frame_bytes or len(b) < 2:
                    conn.enqueue(error_msg(ErrorCode.PROTOCOL_VIOLATION, "binary frame before room.join or bad size"))
                    continue
                kind, payload = b[0], b[1:]
                if kind == BIN_LANDMARKS:
                    if conn.inflight_landmarks > s.ws_max_inflight_landmark_batches:
                        conn.enqueue(error_msg(ErrorCode.BACKPRESSURE, "slow down"))
                        continue
                    try:
                        batch = unpack(payload)
                    except InvalidFrame as e:
                        conn.enqueue(error_msg(ErrorCode.PROTOCOL_VIOLATION, str(e)))
                        continue
                    await session.on_landmarks(batch)
                elif kind == BIN_AUDIO:
                    await session.on_audio(payload)
                continue

            try:
                env, msg = parse_client(m["text"])
            except ProtocolError as e:
                conn.enqueue(error_msg(e.code, e.message))
                continue
            conn.last_peer_seq = env.seq
            WS_MESSAGES.labels("in", msg.type).inc()
            match msg.type:
                case "room.join":
                    async with transaction(principal) as dbc:
                        mem = await join_room.membership(dbc, principal, UUID(msg.room_id))
                        state = await join_room.room_state(dbc, UUID(msg.room_id)) if mem else None
                    if mem is None or state is None:
                        conn.enqueue(error_msg(ErrorCode.NOT_A_MEMBER, "join via REST first"))
                        continue
                    room = msg.room_id
                    conn.room_id, conn.member_id = room, str(mem["id"])
                    conn.role, conn.modality_out = mem["role"], mem["modality_out"]
                    app.state.local.add(room, conn)
                    await app.state.bus.join(room)
                    await session.attach(
                        room_id=UUID(room),
                        member_id=str(mem["id"]),
                        role=mem["role"],
                        modality_out=mem["modality_out"],
                        room_info=state["room"],
                    )
                    await resume.remember_owner(app.state.redis, session_id, principal.subject, room)
                    await _broadcast_state(ws, room)
                case "text.send":
                    ok, retry = await limiter.allow(f"ws_text:{principal.subject}", *POLICIES["ws_text"])
                    if not ok:
                        conn.enqueue(error_msg(ErrorCode.RATE_LIMITED, "slow down", retry))
                        continue
                    if room is None:
                        conn.enqueue(error_msg(ErrorCode.NOT_A_MEMBER, "join a room first"))
                        continue
                    await session.on_text(msg.text, msg.lang)
                case "sign.start":
                    session.set_signing_prefs(msg.fps, msg.dominant_hand)
                    if room and conn.member_id:
                        await presence.signing(app.state.bus, room, conn.member_id, True)
                case "sign.stop":
                    await session.end_signing()
                    if room and conn.member_id:
                        await presence.signing(app.state.bus, room, conn.member_id, False)
                case "speech.start":
                    session.start_speech(msg.lang_hint)
                case "speech.stop":
                    await session.end_speech()
                case "clarify.answer":
                    await session.on_clarify_answer(msg.utterance_id, msg.choice, msg.retry)
                case "feedback":
                    if principal.user_id is not None:
                        async with transaction(principal) as dbc:
                            await record_feedback.execute(
                                dbc,
                                principal.user_id,
                                msg.utterance_id,
                                msg.kind,
                                msg.corrected_glosses,
                                msg.corrected_text,
                            )
                case "typing":
                    if room and conn.member_id:
                        await presence.typing(app.state.bus, room, conn.member_id, msg.active)
                case "ping":
                    conn.enqueue(server_msg("pong", {"t": msg.t}))
                case "auth.refresh":
                    try:
                        newp = get_jwt().decode(msg.token)
                    except Exception:
                        await ws.close(code=int(ErrorCode.AUTH_REQUIRED))
                        break
                    if newp.subject != principal.subject:
                        await ws.close(code=int(ErrorCode.AUTH_REQUIRED))
                        break
                    principal = newp
                    session.principal = newp
                case "device.event" | "device.capabilities":
                    await session.on_device_message(msg.model_dump())
                case _:
                    conn.enqueue(error_msg(ErrorCode.PROTOCOL_VIOLATION, f"unexpected {msg.type}"))
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws_handler_error")
    finally:
        wd.cancel()
        writer.cancel()
        WS_CONNECTIONS.dec()
        for _, text in list(conn.replay):
            seq = int(text.split('"seq":', 1)[1].split(",", 1)[0])
            await resume.save(app.state.redis, session_id, seq, text)
        app.state.local.remove(room, conn)
        if room:
            await app.state.bus.leave_if_empty(room)
            try:
                await _broadcast_state(ws, room)
            except Exception:
                log.exception("room_state_broadcast_failed")
        await session.close()


def _unused(_: Any) -> None:  # keeps type checkers quiet about the dynamic match arms
    return None
