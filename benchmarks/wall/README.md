# Browser wall validation — 2026-09-08

The live test used Chromium with **two real browser RTCPeerConnections**, local
CPU Kokoro synthesis, and the SoulX GPU worker. Source avatar: MuseTalk's certified
`idle_active_listening.mp4`. Profile: native 480×832, four steps, 25-fps target,
batch one, two active-call slots, source idle, eager execution, optimized real
RoPE, staged memory transfers, veryfast H264. The unrelated OmniVoice service
remained running on the same RTX 4070.

- Broadcast one freshly synthesized Kokoro WAV to both calls.
- Both completed the speech turn; a second text/voice synthesis and turn went to
  peer two, using its original peer and tracks (one negotiation per call).
- Browser stats reported H264 video and Opus audio on both peers. At the final
  snapshot both reported zero RTP packet loss and zero dropped video frames.
- Audio selection and narrow-screen layout worked; no browser page errors.
- Stop released all calls and GPU states (`calls=0`, `gpu_states=0`). The local
  service was left running, with no test peers connected.

This validates integration, **not smooth real-time concurrent speech**. The
memory-constrained staged profile was slow: the first broadcast's first sender
media took 9.63s / 4.63s, and the completed turns accumulated 26.40s / 32.80s of
speech audio holds across the two peers (peer two includes the second turn).
Near-25 browser FPS while idle does not establish neural throughput. Warm Kokoro
synthesis for the second turn took 2.66s for 3.38s of audio.

Evidence:

- [Full live counters and cleanup](live-check.json)
- [Two real peers after broadcast](live-two-peers.png)
- [Second turn on the existing peer](live-second-turn.png)
- [Mobile layout](live-mobile.png)

These screenshots precede the final desktop refinement that places setup and TTS
panels side by side. The final layout is also covered by the Chromium regression.

Automated regressions: 34 affected server/TTS tests passed, plus the opt-in
three-peer Chromium lifecycle regression. The latter uses deterministic CPU
frames/speech with real ICE/DTLS/SRTP/H264/Opus and includes lost-response retries,
independent seeds, per-tile speech, interruptions, and Stop during pending create.
Run instructions are in [WEBRTC_WALL.md](../../WEBRTC_WALL.md).
