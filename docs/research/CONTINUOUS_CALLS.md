# First/last frames and invisible turn boundaries

## Answer

**Native SoulX Lite does not expose exact first-frame and last-frame pixel controls.** It accepts a conditioning portrait and carries recent motion latents between chunks. That supports continuous-looking generation; it does not guarantee that independently generated clips return to the same canonical pose or match each other exactly.

The current MuseTalk assets have a stronger source-level contract: their first six and last six decoded frames are the same canonical image. This pass independently decoded the six physical close-up videos and checked all 30 directed non-self endpoint pairs. That is a property of the certified source bank, not of arbitrary generative output or lossy received video.

## Four different continuity guarantees

| Level | What it means | Current evidence |
| --- | --- | --- |
| Transport | Same peer/senders; continuous media timestamps through switches | Source-bank relay passed one persistent H264/Opus session |
| Source pixels | Outgoing canonical source frame equals incoming frame | All six source handles and 30 directed pairs match exactly |
| Generated motion | Head, mouth, gaze, crop and lighting evolve naturally | Native continuous SoulX recordings look coherent in sampled frames; no universal guarantee |
| Conversation | Multiple newly arriving turns, silence and interruptions share a call clock and correct ownership | MuseTalk implements this; current SoulX clip API does not |

One cannot infer the fourth row from a ten-second video or from a packet-loss-free connection. Even pixel-identical endpoints can still reveal a join if velocity, eye state or lighting changes immediately around them. A freeze is pixel-continuous but often perceptually obvious.

## What control exists in the model

Reference conditioning determines identity/composition approximately. Initial history is the first reference latent, not a protected output pixel frame. The current engine discards the first nine decoded frames, so the first **displayed** frame is already generated. Choosing the last emitted index controls duration, not its pose. There is no `end_image`, `last_frame`, explicit pose trajectory, or trained endpoint-inpainting API in the installed pipeline/RTC service.

Within one generation state, corrected output frames `[-9:]` are re-encoded and passed into the next chunk. Keep that state and its RNG across audio segments to exploit the model's existing continuity mechanism. Calling `prepare()` again, reseeding or resetting the person/history for every utterance returns to reference-conditioned startup rather than continuing what the caller last saw.

Overwriting the displayed first/last image with a canonical source frame is possible in application code, but it can create a head snap, closed mouth over audible speech, or jump at the next adjacent frame. It is not native control. Adding a target latent mask/last-image constraint changes the denoising problem and may need training or fine-tuning; treat it as research, not an available switch.

## What this pass measured

The exact original close-up bank was sent at 480×832/24 FPS through a persistent local WebRTC peer: **1350/1350 frames**, 56.25 seconds, five sequential handoffs, one negotiation, no video track replacement, stable single audio/video SSRCs and zero reported RTP packet loss. All 30 source pairs were decoded/offline-audited; only the five sequential handoffs were transmitted in this particular replay. Received seam MAE was ~0.064–0.155 RGB levels rather than zero, consistent with lossy H264. See [base-bank JSON](../../benchmarks/migration/base-bank-webrtc.json).

Native SoulX recordings used the canonical first frame extracted from that **same** production bank, centered/cropped by the existing512 workflow. They do not reuse its complete head-motion trajectory. The longer input is one preassembled 23.05-second waveform: 3.575 seconds speech, two seconds silence, 17.475 seconds speech. One state generated all577 output frames. There was no second live upload.

The [recording analysis](../../benchmarks/migration/recording-boundaries.json) found:

- Continuous23s adjacent-frame RGB MAE median1.18, p95 2.73; chunk-boundary maximum3.06. Native10s chunk-boundary maximum2.59. Chunk joins were not the largest changes in either recording.
- Independent10s→23s and23s→10s endpoint differences were4.27 and3.95 respectively; neither was exact.
- Received first/last frames differed from the cropped conditioning image by ~3.42–4.16 RGB levels.

These are unaligned full-frame differences after WebRTC and recorder encoding. Different audio/seeds and legitimate mouth motion are confounders. They show that these particular decoded clip endpoints are unequal; they do not isolate the reset's causal effect or establish a perceptual threshold. No blinded review, SyncNet, optical-flow or landmark study was completed in this pass.

## Proposed persistent-call architecture — not implemented here

Separate the **call** from an **utterance**. Create one `RTCPeerConnection`, one persistent video sender and one persistent audio sender when the call starts. Do not re-offer, replace tracks, reset timestamps or rebuild the client video element for each response.

```text
IDLE → ARMED(first audio + first video ready) → SPEAKING → DRAINING → IDLE
                    ↑                            │
                    └──── next queued turn ──────┘
SPEAKING / ARMED → INTERRUPT(epoch changes) → bounded settle → IDLE
```

Suggested call state: approved portrait/asset hashes and model profile; private motion/RNG state; audio ring with absolute sample indices; output-frame cursor; selected pose/physical variant; bounded chunk buffers; call/turn/generation IDs; cancellation epoch; generated/sent/acknowledged media horizons. Immutable models remain shared in one GPU owner.

Suggested new API contract (proposal, existing routes unchanged):

- Create call separately from uploading a clip.
- Append a turn using idempotent `turn_id`, monotonic sequence and exact PCM/sample-rate metadata. Same ID + same payload is a retry; same ID + different payload is a conflict.
- Distinguish `accepted`, `first_media`, `generation_complete`, `audio_playout_complete`, `turn_complete` and `interrupted`. HTTP acceptance must not restart the microphone or release idle.
- Cancel by epoch/ownership token. Late callbacks from the old turn cannot enqueue frames, change the pose, finish the next turn or reset its clock.
- Close idempotently, drain/release only owned state, and retain healthy active calls during worker scale-down.

### One media timeline

Use integer sample/frame counters: `video_pts = round(call_frame_index×90000/output_fps)`, audio PTS in48kHz samples, normally20ms/960-sample packets. Per-turn media offsets map into that same call timeline. Continuous silence packets keep the audio sender alive during idle.

Adapt MuseTalk's tested `VideoSyncClock`, persistent audio-source switching and forward-only initial alignment logic, not its older adaptive audio skipping or per-turn audio-track replacement. Stage the neutral frame before audible EOF; return to idle only after the audio **playout horizon** and required video horizon have both been reached. Generation completion is often seconds earlier. Server RTP send completion still is not proof of what a remote device has audibly played; instrument receiver timing and bound its jitter-buffer contribution.

At24FPS one frame is41.67ms; at25FPS it is40ms. The final partial audio packet/video frame needs explicit padding accounting. Don't drop the last phoneme just to force matching container durations. At23.05 seconds, the captured25FPS video legitimately has577 frames/23.08 seconds; audio packets extend to23.06 seconds. Timestamp monotonicity, useful samples and audible completion must be checked separately.

### State must follow displayed history, not only generated history

Current bounded queues can hold several chunks ahead. If a turn is interrupted, the model's latest motion prefix may describe a head pose the receiver never displayed. Reusing that state after discarding queued video causes a visual jump.

Proposed fix: keep a small ring of motion/RNG checkpoints associated with chunk media ranges; commit history at the played boundary. On interruption, roll back to a valid displayed-history checkpoint or explicitly recondition from the last displayed frame window. Partial-chunk interruptions require a defined re-encode policy. Preserve audio-history sample offsets, and ignore stale worker completions via the epoch. Do not assume a deep copy of every model is needed; checkpoint only mutable session state.

### Cheap idle versus model-state continuity

Option A: keep SoulX generating speech and silence in one state. This best uses native recurrence, but spends GPU work during idle and has no exact neutral endpoint.

Option B: replay the certified MuseTalk bank during idle, render speech with SoulX, then bridge back. This saves idle compute and reuses approved gestures. It is **not yet seamless**: native SoulX is square and creates different full-frame motion. At resume, reconditioning on the last observed nine frames needs common geometry/cadence and validation; at speech end, a short settle/bridge must finish after audible EOF. A fixed crop or crossfade cannot guarantee anatomical alignment.

Option C: keep the source video as the background and compose only a SoulX-derived face/mouth region. This aims at deterministic source motion, but introduces segmentation, alignment, identity/scale matching, color seams and recurrent conditioning changes. MuseTalk's failed single-neutral-geometry experiment already shows why reusing one face mask for moving poses is unsafe. It is a new renderer, not “SoulX with the same base video” at no cost.

For current production, retain MuseTalk's prepared per-pose geometry and certified source bank. For SoulX research, implement/test Option A's persistent call and state-ownership contract first; evaluate B separately if idle compute is too expensive. Do not promise Option C without an aligned quality prototype.

## Seamless-call acceptance tests

Use the approved close-up bank, V14/V15 and existing English fixtures, then multiple portraits and varied phonemes. Require:

- One persistent peer/audio/video sender over many turns; no unexpected negotiation, track replacement, SSRC reset or backward PTS.
- All30 directed source edges, six loops, live↔idle and interrupted boundaries—not only the easiest sequential order.
- Frame windows around each boundary checked for exact source handles, landmark/face-region residuals, motion velocity, gaze/blink state, color shift and mouth/audio phase.
- No stale completions, duplicate retry playback or microphone restart before audible EOF. Test rapid double-submit, cancellation during upload, generation and draining, and a slow receiver.
- 30-minute calls, network jitter/loss and real mobile/TURN clients; bounded memory and no growing audio/video drift.
- Human normal-speed playback approval. Numerical continuity is necessary but not sufficient for “the caller cannot tell.”

The transport control and short continuous native recording pass their limited checks. The full interactive acceptance suite is still open.
