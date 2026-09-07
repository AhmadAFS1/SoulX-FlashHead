import numpy as np

from soulx_rtc.check_recorded_audio import scores


def test_correspondence_rejects_silence_roundoff_and_finds_actual_waveform():
    x=np.random.default_rng(42).normal(size=1600)*.1
    y=np.zeros(16000)
    y[4321:4321+len(x)]=x*.8
    result=scores(y,x)
    assert int(np.argmax(result))==4321
    assert abs(result[4321]-1)<1e-10
    assert np.isneginf(result[:2000]).all()
    assert np.max(result)<=1+1e-10
