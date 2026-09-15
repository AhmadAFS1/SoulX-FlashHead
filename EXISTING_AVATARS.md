# Existing avatars in the WebRTC wall

> **GPU provenance — local runs:** NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible). This applies to the local SoulX/MuseTalk inference and receiver tests described here; CPU-only checks do not establish GPU performance. Historical or upstream results on other GPUs retain their separate attribution. See [GPU run provenance](docs/research/GPU_RUN_PROVENANCE.md) for dates, evidence, and attribution limits.

Enter the API token, click **Refresh server**, select **Avatar**, then click
**Create + connect wall**. The selection applies to all new wall peers. Stop and
release the wall before selecting another avatar. Tiles show their avatar name.

Authenticated `GET /avatars` lists server-owned IDs; pass `avatar_id` to
`POST /calls`. The default `--avatar-root` is
`/workspace/MuseTalk/assets/ltx23_pose_banks`. Each bank contributes one certified
idle video, preferring `idle_active_listening.mp4`, then `active_listening.mp4`,
then `neutral_resting.mp4`. Raw candidates and escaping symlinks are excluded.
Clients cannot provide filesystem paths. Restart to reload the catalog.
The server default avatar remains available. Each selected video's first frame
becomes that call's model reference and canonical return-to-idle target.

This is asset reuse, not MuseTalk latent-cache compatibility. Our SoulX engine
caches up to eight prepared templates in memory using image content, dimensions
and steps as the key. The cache is lost on process restart. Each call has private
motion state and random generator. There is no persistent avatar-building API
equivalent to MuseTalk's saved frames, masks, coordinates and latent tensors.
New speech still needs generation; avatar caching does not solve staged-memory
underproduction. The earlier native test needed approximately 2.8–3.2 seconds
per 24-frame chunk, versus 0.96 seconds of playback at 25 FPS.

Validation: `tests/test_avatars.py` checks approved discovery and rejects unknown
IDs, filesystem paths and escaping symlinks. `tests/test_calls.py` selects a
catalog avatar over HTTP, verifies returned identity, and exercises real
H.264/Opus calls and boundary continuity using the selected idle video.
