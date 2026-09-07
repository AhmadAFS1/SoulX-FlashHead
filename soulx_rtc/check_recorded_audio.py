"""Waveform correspondence, not a perceptual lip-sync or receiver-ACK score."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
from scipy.signal import correlate


def scores(recorded, reference):
    y=np.asarray(recorded,np.float64)
    x=np.asarray(reference,np.float64).copy()
    x-=x.mean()
    n=len(x)
    energy=float(x@x)
    if energy<=1e-12 or len(y)<n:
        raise ValueError("Need a non-silent reference and a longer recording")
    sums=np.r_[0,np.cumsum(y)]
    squares=np.r_[0,np.cumsum(y*y)]
    windows=np.maximum(0,squares[n:]-squares[:-n]-(sums[n:]-sums[:-n])**2/n)
    result=np.full(len(windows),-np.inf)
    # FFT roundoff divided by near-zero silence energy creates false peaks.
    valid=windows>energy*.001
    numerator=correlate(y,x,mode="valid",method="fft")
    result[valid]=numerator[valid]/np.sqrt(windows[valid]*energy)
    return result


def main():
    from .server import decode_audio
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recording",required=True)
    ap.add_argument("--reference",required=True)
    ap.add_argument("--seconds",type=float,default=3)
    ap.add_argument("--matches",type=int,default=1)
    ap.add_argument("--output",required=True)
    args=ap.parse_args()
    if not 0<args.seconds<=30 or args.matches<1:
        ap.error("Require seconds in (0,30] and positive matches")
    output=Path(args.output)
    if output.exists():
        ap.error("Use a fresh evidence path")
    raw=subprocess.check_output(["ffmpeg","-v","error","-threads","2","-i",args.recording,
        "-map","0:a:0","-ar","16000","-ac","1","-f","f32le","-"])
    y=np.frombuffer(raw,np.float32)
    x=decode_audio(args.reference,args.seconds)
    correlations=scores(y,x)
    matches=[]
    for _ in range(args.matches):
        i=int(np.argmax(correlations))
        if not np.isfinite(correlations[i]):
            raise ValueError("Not enough non-overlapping non-silent candidate windows")
        segments=[]
        for offset in range(0,len(x)-7999,8000):
            lo=max(0,i+offset-2400)
            local=scores(y[lo:min(len(y),i+offset+10400)],x[offset:offset+8000])
            j=int(np.argmax(local))
            if np.isfinite(local[j]):
                segments.append(dict(reference_start_s=offset/16000,
                    inferred_turn_start_s=(lo+j-offset)/16000,correlation=float(local[j])))
        matches.append(dict(start_s=i/16000,correlation=float(correlations[i]),segments=segments))
        correlations[max(0,i-len(x)):min(len(correlations),i+len(x))]=-np.inf
    result=dict(method=__doc__,config=vars(args),sample_rate=16000,
        recording_sha256=hashlib.sha256(Path(args.recording).read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        matches=sorted(matches,key=lambda row:row["start_s"]),
        note="Normalized cross-correlation after lossy Opus/AAC and resampling. Low-energy windows excluded. No automatic quality pass; changing segment offsets can reveal inserted silence or timing drift.")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
