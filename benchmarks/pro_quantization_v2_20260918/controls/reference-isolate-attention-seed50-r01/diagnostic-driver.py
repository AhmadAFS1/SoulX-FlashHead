import sys
from benchmarks.pro_quantization_v2_20260918 import run
from flash_head.src.modules import flash_head_model as m
import soulx_rtc.pro_attention_backends as a
m.SAGE_ATTN_AVAILABLE=False
m.FLASH_ATTN_3_AVAILABLE=False
run.install_self_attention_backend=lambda model,backend: ['diagnostic: original dispatcher']
a.pin_cross_attention_flash2=lambda model: {'backend':'original_flash2_dispatcher','diagnostic':True}
run.main()
