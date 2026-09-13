import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from soulx_rtc.idle_cache import IdleCache
from soulx_rtc.calls import IdleVideo
from soulx_rtc.shared_chunks import SharedChunks


@pytest.mark.parametrize('size',[(77,103),(100,77),(511,1024)])
def test_portrait_crop_matches_model_without_parent_torch_dependency(tmp_path,size):
    from PIL import Image
    from soulx_rtc.avatars import portrait_anchor
    from flash_head.utils.utils import resize_and_centercrop
    image=Image.fromarray(np.random.default_rng(14).integers(0,256,(size[1],size[0],3),dtype=np.uint8))
    path=tmp_path/'portrait.png';image.save(path)
    expected=resize_and_centercrop(image,(576,320))[0,:,0].permute(1,2,0).numpy()
    assert np.array_equal(expected,portrait_anchor(path,320,576))


def clip_file(path,level=10):
    with av.open(str(path),'w') as out:
        s=out.add_stream('ffv1',rate=24);s.width=s.height=64;s.pix_fmt='bgr0'
        for i in range(8):
            frame=av.VideoFrame.from_ndarray(np.full((64,64,3),level+i,np.uint8),format='rgb24')
            for p in s.encode(frame):out.mux(p)
        for p in s.encode():out.mux(p)
    return str(path)


def test_cached_pixels_loop_rewind_and_independent_peers(tmp_path):
    path=clip_file(tmp_path/'a.mkv');cache=IdleCache(8*64*64*3)
    a=cache.acquire(path,64,64);b=cache.acquire(path,64,64)
    assert a is b and a.users==2
    old=IdleVideo(path,64,64,25,a.frames[0])
    one=IdleVideo(path,64,64,25,a.frames[0],cache=cache,clip=a)
    two=IdleVideo(path,64,64,25,a.frames[0],cache=cache,clip=b)
    try:
        for index in [0,1,8,9,12,50,1,0]:
            assert np.array_equal(old.next(index),one.next(index))
            assert np.array_equal(two.next(0),a.frames[0])
        with pytest.raises(ValueError):a.frames[0].setflags(write=True)
    finally:
        old.close();one.close();one.close();two.close()
    assert cache.stats()['leases']==0 and cache.stats()['hits']==1


def test_budget_pinning_eviction_and_content_invalidation(tmp_path):
    a=clip_file(tmp_path/'a.mkv');b=clip_file(tmp_path/'b.mkv',50)
    cache=IdleCache(8*64*64*3)
    first=cache.acquire(a,64,64)
    assert cache.acquire(b,64,64) is None  # Pinned first clip cannot be evicted.
    assert cache.stats()['reserved_bytes']==0
    cache.release(first)
    second=cache.acquire(b,64,64)
    assert second is not None and cache.stats()['entries']==1
    cache.release(second)
    clip_file(Path(b),80)
    replacement=cache.acquire(b,64,64)
    assert not np.array_equal(second.frames[0],replacement.frames[0])
    cache.release(replacement)
    tiny=IdleCache(100)
    assert tiny.acquire(a,64,64) is None
    assert tiny.stats()['bytes']==tiny.stats()['reserved_bytes']==0


def test_parallel_cache_load_is_deduplicated(tmp_path):
    path=clip_file(tmp_path/'a.mkv');cache=IdleCache(1024**2)
    with ThreadPoolExecutor(4) as pool:
        clips=list(pool.map(lambda _:cache.acquire(path,64,64),range(8)))
    assert len({id(c) for c in clips})==1 and cache.stats()['misses']==1
    assert cache.stats()['leases']==8
    for c in clips:cache.release(c)


def test_shared_slot_never_aliases_retained_frames_and_unlinks():
    owner=SharedChunks((2,24,64,64,3));attached=SharedChunks(**owner.spec())
    name=owner.shm.name
    try:
        source=[np.full((24,64,64,3),17,np.uint8),np.full((5,64,64,3),91,np.uint8)]
        counts=attached.write(source);retained=owner.read(counts)
        attached.write([np.zeros_like(source[0])])
        assert all(np.array_equal(a,b) for a,b in zip(source,retained))
        with pytest.raises(ValueError):owner.read([25])
        with pytest.raises(ValueError):attached.write([np.ones((25,64,64,3),np.uint8)])
        with pytest.raises(ValueError):attached.write([np.ones((1,64,64,3),np.float32)])
    finally:attached.close();owner.close();owner.close()
    from multiprocessing.shared_memory import SharedMemory
    with pytest.raises(FileNotFoundError):SharedMemory(name=name)


def test_cancelled_cache_acquire_releases_late_lease(tmp_path):
    import threading
    from soulx_rtc.server import Service
    path=clip_file(tmp_path/'a.mkv')
    async def run():
        service=Service(SimpleNamespace())
        original=service.idle_cache.acquire
        entered,finish=threading.Event(),threading.Event()
        def slow(*args):
            entered.set();finish.wait(5)
            return original(*args)
        service.idle_cache.acquire=slow
        task=asyncio.create_task(service.acquire_idle(path,64,64))
        await asyncio.to_thread(entered.wait,5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        finish.set()
        # Barrier behind the serialized loader, then let its completion callback run.
        await asyncio.get_running_loop().run_in_executor(service.idle_executor,lambda:None)
        await asyncio.sleep(.01)
        assert service.idle_cache.stats()['leases']==0
        service.executor.shutdown();service.media_executor.shutdown();service.idle_executor.shutdown()
    asyncio.run(run())


def test_startup_freeze_keeps_new_cycles_collectable():
    import subprocess,sys
    program='''
import gc, weakref
gc.collect(); gc.freeze()
class CallCycle: pass
call=CallCycle(); call.self=call
reference=weakref.ref(call)
del call
gc.collect()
assert reference() is None
assert gc.isenabled()
gc.unfreeze()
'''
    subprocess.run([sys.executable,'-c',program],check=True)
