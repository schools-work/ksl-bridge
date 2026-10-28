# Binary WebSocket frames (v1)

Every binary frame starts with **one type byte**.

| Byte | Direction | Payload |
|---|---|---|
| `0x01` | client → server | `KSLB` landmark batch (below) |
| `0x02` | client → server | Audio: PCM16 little-endian, mono, 16 kHz; 20 ms per frame (640 bytes) |
| `0x03` | server → client | Pose sequence for the avatar: same 107-point layout, float16, preceded by a JSON `sign.render` message carrying `utterance_id`, `n_frames`, `fps` |
| `0x04` | server → client | TTS audio chunk (PCM16 LE mono; sample rate in the preceding `speech.audio` JSON message) |

## `KSLB` landmark batch (little-endian)

| Offset | Type | Field |
|---|---|---|
| 0 | `char[4]` | magic `KSLB` |
| 4 | `u8` | spec version (`1`) |
| 5 | `u8` | flags (reserved, 0) |
| 6 | `u16` | `n_frames` (clients send 4) |
| 8 | `u32` | `seq_start`: client frame counter of the first frame |
| 12 | `u64` | `t0_ms`: client monotonic timestamp of the first frame |
| 20 | `f32` | measured fps |
| 24 | `f16[n_frames × 107 × 3]` | coordinates (x, y in [0,1] image-normalised, origin top-left; z relative); missing points written as 0 |
| … | `bits[n_frames × 107]` | presence mask, packed MSB-first, padded to a byte |

**Rules:** front-camera frames are sent **unmirrored**. The server validates the magic, version, size (≤ `KSL_WS_MAX_FRAME_BYTES`) and finite values before decoding.
