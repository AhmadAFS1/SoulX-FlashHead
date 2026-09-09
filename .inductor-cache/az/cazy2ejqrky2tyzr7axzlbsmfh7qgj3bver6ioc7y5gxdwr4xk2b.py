
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'in_ptr5': '*bf16', 'in_ptr6': '*bf16', 'in_ptr7': '*fp32', 'in_ptr8': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sub_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 16, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_sub_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1497600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex % 768)
    x2 = xindex // 768
    x0 = (xindex % 64)
    tmp0 = tl.load(in_ptr0 + (2*x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (2*x0 + 128*x2), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr0 + (1 + 2*x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr1 + (1 + 2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (1 + 2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp26 = tl.load(in_ptr4 + (1 + 2*x0 + 128*x2), xmask, eviction_policy='evict_last')
    tmp32 = tl.load(in_ptr5 + (2*x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp33 = tl.load(in_ptr6 + (2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp36 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp46 = tl.load(in_ptr5 + (1 + 2*x3), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tl.load(in_ptr6 + (1 + 2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp52 = tl.load(in_ptr8 + (1 + 2*x4), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = 1536.0
    tmp6 = (tmp4 / tmp5)
    tmp7 = 1e-06
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp16 = tmp14 * tmp15
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp9
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp28 = tmp16 - tmp27
    tmp29 = tmp14 * tmp26
    tmp30 = tmp25 * tmp15
    tmp31 = tmp29 + tmp30
    tmp34 = tmp32 + tmp33
    tmp35 = tmp34.to(tl.float32)
    tmp37 = (tmp36 / tmp5)
    tmp38 = tmp37 + tmp7
    tmp39 = libdevice.rsqrt(tmp38)
    tmp40 = tmp35 * tmp39
    tmp41 = tmp40.to(tl.float32)
    tmp43 = tmp41 * tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp44 * tmp15
    tmp48 = tmp46 + tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp50 = tmp49 * tmp39
    tmp51 = tmp50.to(tl.float32)
    tmp53 = tmp51 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp55 = tmp54 * tmp26
    tmp56 = tmp45 - tmp55
    tmp57 = tmp44 * tmp26
    tmp58 = tmp54 * tmp15
    tmp59 = tmp57 + tmp58
    tl.store(out_ptr0 + (x3), tmp28, xmask)
    tl.store(out_ptr1 + (x3), tmp31, xmask)
    tl.store(out_ptr2 + (x3), tmp56, xmask)
    tl.store(out_ptr3 + (x3), tmp59, xmask)
