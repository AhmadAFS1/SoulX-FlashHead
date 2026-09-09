import asyncio
import time

import av
import numpy as np

from soulx_rtc.boundaries import Boundaries
from soulx_rtc.calls import Call, CallVideoTrack, IdleVideo, Turn


def test_exact_endpoints_and_epoch_cancellation():
    b=Boundaries()
    anchor=np.full((64,64,3),19,np.uint8)
    target=np.full_like(anchor,201)
    b.begin('in',anchor,'a',0)
    frames=[b.apply(target,0,i*3600) for i in range(4)]
    assert np.array_equal(frames[0],anchor)
    assert np.array_equal(frames[-1],target)
    assert [int(f[0,0,0]) for f in frames]==[19,64,156,201]
    assert b.pending is None
    b.begin('out',target,'a',0)
    assert np.array_equal(b.apply(anchor,1,18000),anchor)
    assert b.events[-1]['status']=='cancelled'
    assert b.pending is None


def test_track_anchor_starvation_and_canonical_return():
    async def run():
        canonical=np.full((64,64,3),10,np.uint8)
        displayed=np.full_like(canonical,70)
        speech=np.full_like(canonical,190)
        c=Call('c',None,25,IdleVideo(None,64,64,25,canonical))
        c.sent_history.append(displayed)
        t=Turn('t','d',np.zeros(0,np.float32),np.zeros(0,np.int16),4,
               start_frame=0,start_sample=0)
        c.active=t
        c.current_frames.append(speech)
        track=CallVideoTrack(c)
        first=await track.recv()
        assert np.array_equal(first.to_ndarray(format='rgb24'),displayed)
        assert c.boundaries.pending['index']==1
        held=await track.recv()
        assert np.array_equal(held.to_ndarray(format='rgb24'),displayed)
        assert c.boundaries.pending['index']==1
        c.current_frames.extend([speech]*3)
        for _ in range(3):
            last=await track.recv()
        assert np.array_equal(last.to_ndarray(format='rgb24'),speech)
        assert c.returning_idle and c.active is None
        outro=[await track.recv() for _ in range(4)]
        assert np.array_equal(outro[0].to_ndarray(format='rgb24'),speech)
        assert np.array_equal(outro[-1].to_ndarray(format='rgb24'),canonical)
        assert not c.returning_idle and c.idle_index==0
        pts=[first.pts,held.pts,last.pts]+[f.pts for f in outro]
        assert all(b>a for a,b in zip(pts,pts[1:]))
    asyncio.run(run())


def test_canonical_idle_rewind_with_musetalk_video():
    from pathlib import Path
    import pytest
    path=Path('/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/idle_active_listening.mp4')
    if not path.exists():
        pytest.skip('Deployment MuseTalk avatar fixture not installed')
    with av.open(str(path)) as src:
        anchor=next(src.decode(video=0)).reformat(width=480,height=832,format='rgb24').to_ndarray()
    idle=IdleVideo(str(path),480,832,25,anchor)
    try:
        assert np.array_equal(idle.next(0),anchor)
        idle.next(20)
        assert np.array_equal(idle.next(0),idle.anchor)
        assert not idle.anchor.flags.writeable
    finally:
        idle.close()


def test_concurrent_close_waits_for_gpu_release():
    from types import SimpleNamespace
    from soulx_rtc.calls import CallService
    async def run():
        entered,release=asyncio.Event(),asyncio.Event()
        async def release_state(state):
            entered.set()
            await release.wait()
        service=CallService(SimpleNamespace(worker=SimpleNamespace(release=release_state)))
        c=Call('closing',None,25,IdleVideo(None,64,64,25,np.zeros((64,64,3),np.uint8)))
        service.items[c.id]=c
        first=asyncio.create_task(service.close(c))
        await entered.wait()
        second=asyncio.create_task(service.close(c))
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        assert c.id in service.items
        release.set()
        await asyncio.gather(first,second)
        assert c.id not in service.items
    asyncio.run(run())
