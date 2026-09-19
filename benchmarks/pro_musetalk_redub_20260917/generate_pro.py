"""Reuse the tested official PRO runner, with this experiment's exact inputs."""
import sys
from pathlib import Path
from soulx_rtc.gpu_lease import acquire_gpu_lease
from benchmarks.pro_lite_teeth_20260917 import run_variant

ROOT=Path(__file__).resolve().parent
lease=acquire_gpu_lease()
run_variant.AUDIO=ROOT/'audio-A.wav'
run_variant.SEED=51
sys.argv=[sys.argv[0],'--variant','pro','--output',str(ROOT/'pro'),
          '--reference',str(ROOT/'reference-320x576.png'),'--framing','silence-seed51-first-frame-320x576']
run_variant.main()
