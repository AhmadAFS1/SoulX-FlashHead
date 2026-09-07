"""Explicit low-latency H264 configuration for the pinned aiortc 1.14 sender.

Packetization, keyframe handling and bitrate feedback remain in aiortc. This
changes CPU encoding, not the neural generation rate or resolution.
"""
from fractions import Fraction

import av
from aiortc.codecs.h264 import H264Encoder


class FastH264Encoder(H264Encoder):
    def __init__(self, fps=25, preset="veryfast"):
        super().__init__()
        self.fps, self.preset = fps, preset

    def _encode_frame(self, frame, force_keyframe):
        # Mirror the upstream lifecycle condition so every replacement context
        # receives the requested options, including after bitrate feedback.
        if self.codec and (frame.width != self.codec.width or frame.height != self.codec.height
                or abs(self.target_bitrate-self.codec.bit_rate)/self.codec.bit_rate > .1):
            self.codec = None
            self.buffer_data, self.buffer_pts = b"", None
        if self.codec is None:
            codec = av.CodecContext.create("libx264","w")
            codec.width, codec.height = frame.width, frame.height
            codec.bit_rate, codec.pix_fmt = self.target_bitrate, "yuv420p"
            codec.framerate, codec.time_base = Fraction(self.fps), Fraction(1,90000)
            codec.thread_count = 2
            codec.options = {"level":"31","tune":"zerolatency","preset":self.preset}
            codec.profile = "Baseline"
            self.codec = codec
        yield from super()._encode_frame(frame,force_keyframe)


def install_encoder_factory(fps, preset):
    """Process-scoped opt-in hook; never patch packages on disk."""
    import aiortc
    import aiortc.codecs
    import aiortc.rtcrtpsender
    if aiortc.__version__ != "1.14.0":
        raise RuntimeError("Custom encoder requires the tested aiortc 1.14.0; revalidate before upgrading")
    previous = aiortc.rtcrtpsender.get_encoder
    if previous is not aiortc.codecs.get_encoder:
        raise RuntimeError("An encoder factory is already installed")
    def factory(codec):
        if codec.mimeType.lower()=="video/h264":
            return FastH264Encoder(fps,preset)
        return previous(codec)
    aiortc.rtcrtpsender.get_encoder = factory
    return previous


def sender_encoder_info(pc):
    """Diagnostic only: report the actual pinned sender, not an env request."""
    if pc is None:
        return []
    result=[]
    for sender in pc.getSenders():
        if sender.kind!="video":
            continue
        encoder=getattr(sender,"_RTCRtpSender__encoder",None)
        context=getattr(encoder,"codec",None)
        result.append(dict(encoder=type(encoder).__name__,
            codec=getattr(context,"name",None),threads=getattr(context,"thread_count",None),
            preset=getattr(encoder,"preset","upstream")))
    return result
