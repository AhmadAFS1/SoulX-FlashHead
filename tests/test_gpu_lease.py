import pytest

from soulx_rtc.gpu_lease import acquire_gpu_lease


def test_gpu_owner_is_exclusive_and_released(tmp_path):
    path=tmp_path/"owner.lock"
    with acquire_gpu_lease(path):
        with pytest.raises(RuntimeError,match="Another SoulX"):
            acquire_gpu_lease(path)
    with acquire_gpu_lease(path):
        pass
