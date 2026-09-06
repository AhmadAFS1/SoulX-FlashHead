"""Source-asset control tests. No model weights or GPU required."""
import asyncio
import json
from fractions import Fraction

import av
import numpy as np
import pytest
from PIL import Image

from soulx_rtc.analyze_recordings import inspect
from soulx_rtc.benchmark_rtc import input_provenance
from soulx_rtc.replay_lab import audit_sources, replay, source_paths


def clip(path, value=90, frames=8, fps=24, size=64):
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264", rate=fps)
        stream.width = stream.height = size
        stream.pix_fmt = "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((size, size, 3), value, np.uint8), format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, fps)
            for packet in stream.encode(frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def test_manifest_aliases_and_escape(tmp_path):
    bank = tmp_path / "bank" / "certified"
    bank.mkdir(parents=True)
    clip(bank / "a.mp4")
    clip(bank / "b.mp4")
    manifest = tmp_path / "manifest.json"
    data = {"asset_bundle": "bank", "poses": {
        "idle": {"asset_file": "a.mp4"},
        "listen": {"asset_file": "a.mp4"},
        "talk": {"asset_file": "a.mp4", "variants": [{"asset_file": "b.mp4"}]}}}
    manifest.write_text(json.dumps(data))
    assert [p.name for p in source_paths(manifest, tmp_path)] == ["a.mp4", "b.mp4"]
    data["poses"]["idle"]["asset_file"] = "../../escape.mp4"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        source_paths(manifest, tmp_path)


def test_source_audit(tmp_path):
    paths = [tmp_path / name for name in ("a.mp4", "b.mp4")]
    for path in paths:
        clip(path)
    rows, edges = audit_sources(paths)
    assert [r["start_frame"] for r in rows] == [0, 8]
    assert len(edges) == 2 and all(e["exact"] and e["rgb_mae"] == 0 for e in edges)
    assert rows[0]["first_six_sha256"] == rows[1]["last_six_sha256"]
    assert rows[0]["sha256"] == input_provenance(paths[0])["sha256"]
    clip(paths[1], size=96)
    with pytest.raises(ValueError, match="geometry"):
        audit_sources(paths)
    with pytest.raises(ValueError, match="at least one"):
        audit_sources([])


def test_replay_real_persistent_peer(tmp_path):
    paths = [tmp_path / name for name in ("a.mp4", "b.mp4")]
    for path in paths:
        clip(path)
    result = asyncio.run(replay(paths, tmp_path / "result.json", record=True))
    assert result["passed_transport_control"], result
    assert result["received_frames"] == 16
    assert result["negotiations"] == 1
    assert result["video_track_replacements"] == 0
    with av.open(str(tmp_path / "result.mp4")) as recording:
        assert sum(1 for _ in recording.decode(video=0)) == 16
    audit, endpoints = inspect(tmp_path / "result.mp4", Image.new("RGB", (64, 64), (90, 90, 90)))
    assert audit["frames"] == 16 and audit["pts_monotonic"]
    assert endpoints[0].shape == endpoints[1].shape == (64, 64, 3)
