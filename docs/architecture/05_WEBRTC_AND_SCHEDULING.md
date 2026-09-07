# 5. HTTP, persistent WebRTC calls, scheduling and clocks

[Guide index](README.md) · [Previous](04_ENGINE_AND_MEMORY.md) · [Next: TensorRT and tests](06_TENSORRT_AND_TEST_TOOLING.md)

Primary sources: [server.py](../../soulx_rtc/server.py), [calls.py](../../soulx_rtc/calls.py), [worker.py](../../soulx_rtc/worker.py), [codec.py](../../soulx_rtc/codec.py), [browser client](../../soulx_rtc/index.html). Exact request examples remain in [CONTINUOUS_WEBRTC.md](../../CONTINUOUS_WEBRTC.md).

## Two service workflows

The finite-session API prepares a bounded image/audio generation job and streams its resulting media. The persistent-call API creates a long-lived peer whose video and audio tracks survive multiple speech turns, silence and interruption.

| Operation family | Responsibility |
| --- | --- |
| Health/readiness | Report execution profile, backend state, session counts and recent metrics |
| Finite `/sessions` | Validate uploaded image/audio, create generation state, negotiate/play finite media, release |
| Persistent `/calls` | Create a call anchored to the approved server idle asset |
| Call offer | Negotiate one peer connection and attach stable H264/Opus tracks |
| Call turns | Validate and enqueue audio with a caller-supplied idempotency key |
| Call interruption | Advance epoch, invalidate unsent work, recover from recently sent frames |
| Call metrics/deletion | Report state; close transports and release worker resources |

Do not assume an arbitrary client path is a supported source-avatar selector. The persistent call uses the server-approved idle asset and a seed; uploaded portraits belong to the finite workflow.

## Input and request bounds

The server bounds request size, image pixels and decoded audio characteristics. The finite path limits uploaded images, thumbnails preparation input, validates channel/rate limits, rejects nonfinite audio and limits usable duration. Long finite input can be truncated to its accepted duration; a successful request does not imply the whole uploaded container was generated.

Persistent audio turns are limited to 30 seconds, with four queued turns and a bounded idempotency history. IDs are 1–128 characters and cannot use the reserved idle prefix. Reusing an ID with identical uploaded bytes returns the original turn; different bytes conflict. Byte identity is deliberate: equal decoded audio in differently encoded files is not necessarily the same request hash.

After 2048 recorded turns, the call must rotate rather than growing idempotency state without limit. Detailed turn metrics retain only a recent window (64); global completed counts and recent-latency samples therefore have different coverage.

Completed turns release large audio arrays. The generated-idle synthetic turn is not inserted as a new user idempotency entry on every extension.

## Call lifetime and turn state machine

```text
call created -> reference state ready -> one offer -> connected
                                                    |
                      queued audio -> admitted -> generating
                                                    |
                        first chunk -> A/V armed -> playing
                                                    |
                         generation may finish     | still draining
                                                    |
                         video tail + useful audio complete
                                                    |
                                           turn complete
                                                    |
                                     next turn or idle
interrupt from connected state:
  epoch++ -> invalidate queued/current unsent work -> await in-flight compute
          -> recondition last sent history -> recover on same transports
close:
  invalidate -> close peer -> release worker state -> close idle decoder
```

The per-call control lock serializes control operations; the generation/state lock protects worker mutations. Epoch tagging prevents a completed old GPU future from reintroducing media after interruption.

**A throughput-relevant coupling remains:** the next speech turn is admitted after the current active turn finishes *playout*, not simply after its model state finishes generation. The GPU can be ready to advance while the current turn drains. Generated silence already has bounded continuation/lookahead; queued speech does not have an equivalent split between generation ownership and playout ownership.

Fixing that requires more than removing an `if`: audio append currently requires a fully generated, chunk-aligned state, and media must remain tagged to the correct turn, epoch and sample range.

## Video clock

The video track follows a persistent wall-clock cadence. It does not emit catch-up bursts after missing time slots. It records missed slots separately from sent frames.

When valid generated frames are ready, it emits them in order. While waiting for generation, turn readiness or draining transitions, it may hold the most recent frame. With source-idle policy it decodes the approved source video; with hold policy it stays still; with generated-idle policy it consumes recurrent neural output.

Recent history contains the last nine **sent raw RGB frames**. This is the source for interruption reconditioning. It is not the receiver's post-H264 image history and is not an acknowledgment of client playout.

The finite video track behaves differently: it waits for new productive frames and adjusts its shared timing origin on starvation. Its transport-drain frames must not count as new generated content.

## Audio clock and A/V arming

Persistent audio is mono 48 kHz with 960-sample packets: one 20 ms tick. Its RTP sample counter never resets between turns. Silence fills idle and waiting periods.

When a turn has its first video chunk, the call arms a shared start point, with a small future buffer (120 ms in this implementation) and clock alignment. Actual speech samples are withheld until the gate opens.

During speech, audio is bounded by the sent-video horizon, with an allowed one-packet lead after video starts. If video cannot progress, the transport still emits silence but does not consume the useful speech samples beyond its allowed horizon. The explicit `audio_hold_samples` counter measures these inserted speech-time holds.

A turn is complete only after its rounded video frame count is sent **and** its useful audio samples are consumed. This protects against switching to idle while speech remains, but also makes whole-chunk tail padding and playout coupling visible in turn latency.

These are sender-side clocks. End-to-end lip synchronization, packet loss, jitter buffers, browser decode delay and acoustic playout require receiver-side measurements.

## Scheduling and admission

The finite scheduler favors sessions with less generated lead and older scheduling time, skips full queues and batches compatible states. Persistent-call scheduling runs between finite batches and uses the configured active-call limit.

There is one GPU worker, but **not yet one unified admission/accounting layer** for finite jobs, call speech, generated idle and reference preparation. Finite jobs are not charged against the persistent-call active quota. All tasks eventually serialize on the same worker, so a cold preparation or finite workload can delay a call even when the call-specific limit appears respected.

The current compatible-call microbatch code is an implementation capability, not a proven optimal batch policy. In particular, installed FFN TRT is exact batch one. For recurrent chunks, batching two sessions means two full five-time-position DiT inputs; it is not MuseTalk's two independent face frames.

A useful future policy estimates finish time from measured per-profile batch latency and selects work against the earliest playout deadline. It must balance fairness and batch efficiency, decline batches that cannot meet the oldest deadline, and reserve bounded capacity for preparation without allowing it to block active media indefinitely.

## Idle policies and what they cost

| Policy | GPU work | Continuity implication |
| --- | --- | --- |
| Source | No SoulX inference for idle frames | Cheap motion plate; generated/source seam may change pose or appearance |
| Hold | No ongoing idle inference | Stable identity but visibly frozen |
| Generate | Full recurrent inference on silence | Better trajectory continuity; consumes active rendering capacity |

Source assets are 24-FPS video while a call can send at 25 FPS; source-time indexing can naturally repeat frames. Those repeats are not GPU-generated FPS.

Generated idle appends another 24-frame silence block when its current state is finished, its queue is shallow and no peer is waiting for speech. It yields future extensions when speech is pending, but already generated/playing idle still drains and occupies the active budget. Source or hold idle is the scalable default candidate for many connected calls, subject to visual acceptance of transitions.

Do not attribute all historical held frames to a scheduler defect. The existing call benchmark intentionally waits for completion and adds a configured gap before the next turn. A new queued-turn fixture is needed to measure avoidable holds when speech is already available.

## Interruption and cleanup, step by step

Interruption first invalidates the old epoch and clears queued/current unsent work and active-turn audio. It then waits for any in-flight call generation to leave the state lock. Old completion results are discarded. The worker re-encodes recent sent RGB history and resets the private continuation state.

There is no attempt to cancel a CUDA kernel halfway through execution. There is no exact rewind of future RNG draws. Reconditioning is an explicitly approximate recovery mechanism.

Deletion closes the peer and tracks, waits for state ownership, releases the worker state, closes the source decoder and drops retained frame/audio objects. Disconnected calls have timed cleanup; connected idle calls may persist. Long-lived capacity therefore depends on explicit application lifecycle and turn-history limits.

## Codec and browser boundaries

H264 preference is set before applying the remote description; later changes to preferences are not equivalent negotiation. The fork optionally installs a version-checked `FastH264Encoder` with two threads and the `veryfast` libx264 preset, retaining the upstream path as fallback.

The browser uses one offer and the same tracks through turns. It waits for ICE gathering with a bound and uses configured ICE servers. Local successful DTLS/H264/Opus loopback does not validate mobile NAT traversal, public TURN capacity, TLS deployment or slow cellular receivers.

A shared bearer token is access control, not tenant isolation. Public hosting needs ownership checks for call IDs, rate limits, TLS, trusted origins and a TURN policy. A codec preset, source hash or health flag is not proof that all clients negotiated that codec; verify actual sender/receiver stats.

## Metrics that must remain separate

- `video_sent`: emitted frames, including repeated/idle frames.
- `generated_frames`: new chunk frames, including neural idle and tail padding.
- `completed_useful_video_frames`: useful frames associated with completed speech.
- `held_frames`: repeated imagery while the call remains paced.
- `missed_video_slots` / underruns: missed cadence slots, distinct from holds.
- `audio_hold_samples`: inserted waiting time during speech, not ordinary idle silence.
- Receiver wire FPS, PTS gaps and packet loss: transport/delivery properties.
- First-media time: admission/preparation/generation/buffering from the recorded request boundary, not ASR+LLM+TTS conversational latency.

The next plan makes these explicit release gates rather than treating `transport_pass` as a smooth-call pass.
