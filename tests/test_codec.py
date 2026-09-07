from fractions import Fraction

import av
import numpy as np

from soulx_rtc.codec import FastH264Encoder


def test_fast_h264_packets_decode_and_recreate_on_bitrate_change():
    from aiortc.codecs.h264 import H264Decoder, h264_depayload
    from aiortc.jitterbuffer import JitterFrame
    encoder, decoder = FastH264Encoder(), H264Decoder()
    count = 0
    for index in range(12):
        if index == 6:
            encoder.target_bitrate = 1800000
        rgb=np.full((96,64,3),index*15,np.uint8)
        frame=av.VideoFrame.from_ndarray(rgb,format="rgb24")
        frame.pts,frame.time_base=index*3600,Fraction(1,90000)
        payloads,pts=encoder.encode(frame,force_keyframe=index in (0,6))
        assert pts==index*3600 and payloads
        decoded=decoder.decode(JitterFrame(data=b"".join(h264_depayload(p) for p in payloads),timestamp=pts))
        assert all((f.width,f.height)==(64,96) for f in decoded)
        count+=len(decoded)
        assert encoder.codec.name=="libx264" and encoder.codec.thread_count==2
    assert count==12
