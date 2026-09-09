# AOT ID: ['2_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/nc/cnczem7iexk5a3xzrghf4px3wev5euz5rsxu46woesu62q2cx4ce.py
# Topologically Sorted Source Nodes: [audio], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   audio => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg0_1, torch.bfloat16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1520640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/gu/cguwmfevgfwctdynohcu5bngfesalvobws4of674siwy7f5vvv56.py
# Topologically Sorted Source Nodes: [packed], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   packed => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_1, %view_2, %view_3], 2), kwargs = {})
triton_poi_fused_cat_1 = async_compile.triton('triton_poi_fused_cat_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 9216) % 12)
    x0 = (xindex % 9216)
    x2 = xindex // 110592
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 3, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (46080 + x0 + 9216*(x1) + 368640*x2), tmp4, other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 9, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr0 + (110592 + x0 + 46080*((-3) + x1) + 368640*x2), tmp9, other=0.0).to(tl.float32)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 12, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (387072 + x0 + 9216*((-9) + x1) + 368640*x2), tmp11, other=0.0).to(tl.float32)
    tmp15 = tl.where(tmp9, tmp10, tmp14)
    tmp16 = tl.where(tmp4, tmp5, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/y5/cy5acihvzj3kpeyc4ztmku5xfsgydftlw2m6kouxrusa6ee7otag.py
# Topologically Sorted Source Nodes: [audio_embeds_c], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   audio_embeds_c => cat_1
# Graph fragment:
#   %cat_1 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%view_8, %view_9], 1), kwargs = {})
triton_poi_fused_cat_2 = async_compile.triton('triton_poi_fused_cat_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 512
    x0 = (xindex % 512)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x0), xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full([1], 0, tl.int32)
    tmp9 = triton_helpers.maximum(tmp8, tmp7)
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 5, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = tl.load(in_ptr2 + (x0 + 512*((-1) + x1)), xmask & tmp12, other=0.0).to(tl.float32)
    tmp16 = tl.load(in_ptr3 + (x0), xmask & tmp12, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 0, tl.int32)
    tmp19 = triton_helpers.maximum(tmp18, tmp17)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp12, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp11, tmp21)
    tl.store(out_ptr0 + (x2), tmp22, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/p7/cp7nvjwokflacc5xpgiur3dqo6ogevf53e7e47txbyyxoiv6yhkn.py
# Topologically Sorted Source Nodes: [linear_2, audio_embeds_c_2], Original ATen: [aten.addmm, aten.relu]
# Source node to ATen node mapping:
#   audio_embeds_c_2 => relu_2
#   linear_2 => add_tensor_31
# Graph fragment:
#   %add_tensor_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mm_default_31, %arg6_1), kwargs = {})
#   %relu_2 : [num_users=1] = call_function[target=torch.ops.aten.relu.default](args = (%add_tensor_31,), kwargs = {})
triton_poi_fused_addmm_relu_3 = async_compile.triton('triton_poi_fused_addmm_relu_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_addmm_relu_3', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_addmm_relu_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 512)
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/6t/c6tevbehjme5iyzzzid3nxdq3gji4qpaeltue62gycvprvlm4hez.py
# Topologically Sorted Source Nodes: [context_tokens_1, projected], Original ATen: [aten._to_copy, aten.native_layer_norm]
# Source node to ATen node mapping:
#   context_tokens_1 => convert_element_type_15, var_mean
#   projected => convert_element_type_16
# Graph fragment:
#   %convert_element_type_15 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_11, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %convert_element_type_16 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.bfloat16), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_4 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_4', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 2, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_4(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 32)
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_2 + 1536*x3), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_2 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp11 = tl.load(in_out_ptr0 + (r0_2 + 1536*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp12 = tl.load(in_ptr0 + (r0_2 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp22 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 - tmp5
        tmp16 = 1536.0
        tmp17 = (tmp6 / tmp16)
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = libdevice.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp23 = tmp22.to(tl.float32)
        tmp24 = tmp21 * tmp23
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp24 + tmp26
        tmp28 = tmp27.to(tl.float32)
        tl.store(in_out_ptr0 + (r0_2 + 1536*x3), tmp28, xmask & r0_mask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/gr/cgrip7rl2ljdfrlpaa6odryyyvxhienqsvgn6vdepom23gzqk2gf.py
# Topologically Sorted Source Nodes: [float_1, pow_1, mean, add, rsqrt, mul, to_2, mul_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
# Source node to ATen node mapping:
#   add => add_2
#   float_1 => convert_element_type_20
#   mean => mean
#   mul => mul_2
#   mul_1 => mul_3
#   pow_1 => pow_1
#   rsqrt => rsqrt_1
#   to_2 => convert_element_type_21
# Graph fragment:
#   %convert_element_type_20 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_15, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_20, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_20, %rsqrt_1), kwargs = {})
#   %convert_element_type_21 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_2, torch.bfloat16), kwargs = {})
#   %mul_3 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_21, %arg13_1), kwargs = {})
triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 160
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_out_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 1536.0
        tmp13 = (tmp6 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(in_out_ptr0 + (r0_1 + 1536*x0), tmp20, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 33, 5, 12, 768), (1520640, 46080, 9216, 768, 1))
    assert_size_stride(arg1_1, (512, 46080), (46080, 1))
    assert_size_stride(arg2_1, (512, ), (1, ))
    assert_size_stride(arg3_1, (512, 110592), (110592, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, 512), (512, 1))
    assert_size_stride(arg6_1, (512, ), (1, ))
    assert_size_stride(arg7_1, (49152, 512), (512, 1))
    assert_size_stride(arg8_1, (49152, ), (1, ))
    assert_size_stride(arg9_1, (1536, ), (1, ))
    assert_size_stride(arg10_1, (1536, ), (1, ))
    assert_size_stride(arg11_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg12_1, (1536, ), (1, ))
    assert_size_stride(arg13_1, (1536, ), (1, ))
    assert_size_stride(arg14_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg15_1, (1536, ), (1, ))
    assert_size_stride(arg16_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg17_1, (1536, ), (1, ))
    assert_size_stride(arg18_1, (1536, ), (1, ))
    assert_size_stride(arg19_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg20_1, (1536, ), (1, ))
    assert_size_stride(arg21_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg22_1, (1536, ), (1, ))
    assert_size_stride(arg23_1, (1536, ), (1, ))
    assert_size_stride(arg24_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg25_1, (1536, ), (1, ))
    assert_size_stride(arg26_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg27_1, (1536, ), (1, ))
    assert_size_stride(arg28_1, (1536, ), (1, ))
    assert_size_stride(arg29_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg30_1, (1536, ), (1, ))
    assert_size_stride(arg31_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg32_1, (1536, ), (1, ))
    assert_size_stride(arg33_1, (1536, ), (1, ))
    assert_size_stride(arg34_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg35_1, (1536, ), (1, ))
    assert_size_stride(arg36_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg37_1, (1536, ), (1, ))
    assert_size_stride(arg38_1, (1536, ), (1, ))
    assert_size_stride(arg39_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg40_1, (1536, ), (1, ))
    assert_size_stride(arg41_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg42_1, (1536, ), (1, ))
    assert_size_stride(arg43_1, (1536, ), (1, ))
    assert_size_stride(arg44_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg45_1, (1536, ), (1, ))
    assert_size_stride(arg46_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg47_1, (1536, ), (1, ))
    assert_size_stride(arg48_1, (1536, ), (1, ))
    assert_size_stride(arg49_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg50_1, (1536, ), (1, ))
    assert_size_stride(arg51_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg52_1, (1536, ), (1, ))
    assert_size_stride(arg53_1, (1536, ), (1, ))
    assert_size_stride(arg54_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg55_1, (1536, ), (1, ))
    assert_size_stride(arg56_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg57_1, (1536, ), (1, ))
    assert_size_stride(arg58_1, (1536, ), (1, ))
    assert_size_stride(arg59_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg60_1, (1536, ), (1, ))
    assert_size_stride(arg61_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg62_1, (1536, ), (1, ))
    assert_size_stride(arg63_1, (1536, ), (1, ))
    assert_size_stride(arg64_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg65_1, (1536, ), (1, ))
    assert_size_stride(arg66_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg67_1, (1536, ), (1, ))
    assert_size_stride(arg68_1, (1536, ), (1, ))
    assert_size_stride(arg69_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg70_1, (1536, ), (1, ))
    assert_size_stride(arg71_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg72_1, (1536, ), (1, ))
    assert_size_stride(arg73_1, (1536, ), (1, ))
    assert_size_stride(arg74_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg75_1, (1536, ), (1, ))
    assert_size_stride(arg76_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg77_1, (1536, ), (1, ))
    assert_size_stride(arg78_1, (1536, ), (1, ))
    assert_size_stride(arg79_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg80_1, (1536, ), (1, ))
    assert_size_stride(arg81_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg82_1, (1536, ), (1, ))
    assert_size_stride(arg83_1, (1536, ), (1, ))
    assert_size_stride(arg84_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg85_1, (1536, ), (1, ))
    assert_size_stride(arg86_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg87_1, (1536, ), (1, ))
    assert_size_stride(arg88_1, (1536, ), (1, ))
    assert_size_stride(arg89_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg90_1, (1536, ), (1, ))
    assert_size_stride(arg91_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg92_1, (1536, ), (1, ))
    assert_size_stride(arg93_1, (1536, ), (1, ))
    assert_size_stride(arg94_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg95_1, (1536, ), (1, ))
    assert_size_stride(arg96_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (1536, ), (1, ))
    assert_size_stride(arg99_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg100_1, (1536, ), (1, ))
    assert_size_stride(arg101_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg102_1, (1536, ), (1, ))
    assert_size_stride(arg103_1, (1536, ), (1, ))
    assert_size_stride(arg104_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg105_1, (1536, ), (1, ))
    assert_size_stride(arg106_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg107_1, (1536, ), (1, ))
    assert_size_stride(arg108_1, (1536, ), (1, ))
    assert_size_stride(arg109_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg110_1, (1536, ), (1, ))
    assert_size_stride(arg111_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg112_1, (1536, ), (1, ))
    assert_size_stride(arg113_1, (1536, ), (1, ))
    assert_size_stride(arg114_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg115_1, (1536, ), (1, ))
    assert_size_stride(arg116_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg117_1, (1536, ), (1, ))
    assert_size_stride(arg118_1, (1536, ), (1, ))
    assert_size_stride(arg119_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg120_1, (1536, ), (1, ))
    assert_size_stride(arg121_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg122_1, (1536, ), (1, ))
    assert_size_stride(arg123_1, (1536, ), (1, ))
    assert_size_stride(arg124_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg125_1, (1536, ), (1, ))
    assert_size_stride(arg126_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg127_1, (1536, ), (1, ))
    assert_size_stride(arg128_1, (1536, ), (1, ))
    assert_size_stride(arg129_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg130_1, (1536, ), (1, ))
    assert_size_stride(arg131_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg132_1, (1536, ), (1, ))
    assert_size_stride(arg133_1, (1536, ), (1, ))
    assert_size_stride(arg134_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg135_1, (1536, ), (1, ))
    assert_size_stride(arg136_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg137_1, (1536, ), (1, ))
    assert_size_stride(arg138_1, (1536, ), (1, ))
    assert_size_stride(arg139_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg140_1, (1536, ), (1, ))
    assert_size_stride(arg141_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg142_1, (1536, ), (1, ))
    assert_size_stride(arg143_1, (1536, ), (1, ))
    assert_size_stride(arg144_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg145_1, (1536, ), (1, ))
    assert_size_stride(arg146_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg147_1, (1536, ), (1, ))
    assert_size_stride(arg148_1, (1536, ), (1, ))
    assert_size_stride(arg149_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg150_1, (1536, ), (1, ))
    assert_size_stride(arg151_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg152_1, (1536, ), (1, ))
    assert_size_stride(arg153_1, (1536, ), (1, ))
    assert_size_stride(arg154_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg155_1, (1536, ), (1, ))
    assert_size_stride(arg156_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg157_1, (1536, ), (1, ))
    assert_size_stride(arg158_1, (1536, ), (1, ))
    assert_size_stride(arg159_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg160_1, (1536, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 33, 5, 12, 768), (1520640, 46080, 9216, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [audio], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(arg0_1, buf0, 1520640, stream=stream0)
        del arg0_1
        buf1 = empty_strided_cuda((1, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf0, (1, 46080), (0, 1), 0), reinterpret_tensor(arg1_1, (46080, 512), (1, 46080), 0), out=buf1)
        del arg1_1
        buf2 = empty_strided_cuda((1, 4, 12, 12, 768), (442368, 110592, 9216, 768, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [packed], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_1.run(buf0, buf2, 442368, stream=stream0)
        del buf0
        buf3 = empty_strided_cuda((4, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf2, (4, 110592), (110592, 1), 0), reinterpret_tensor(arg3_1, (110592, 512), (1, 110592), 0), out=buf3)
        del arg3_1
        del buf2
        buf4 = empty_strided_cuda((1, 5, 512), (2560, 512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [audio_embeds_c], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_2.run(buf1, arg2_1, buf3, arg4_1, buf4, 2560, stream=stream0)
        del arg2_1
        del arg4_1
        del buf1
        del buf3
        buf5 = empty_strided_cuda((5, 512), (512, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf4, (5, 512), (512, 1), 0), reinterpret_tensor(arg5_1, (512, 512), (1, 512), 0), out=buf5)
        del arg5_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Topologically Sorted Source Nodes: [linear_2, audio_embeds_c_2], Original ATen: [aten.addmm, aten.relu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_addmm_relu_3.run(buf6, arg6_1, 2560, stream=stream0)
        del arg6_1
        buf7 = empty_strided_cuda((5, 49152), (49152, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_2, audio_embeds_c_2, linear_3], Original ATen: [aten.addmm, aten.relu]
        extern_kernels.mm(buf6, reinterpret_tensor(arg7_1, (512, 49152), (1, 512), 0), out=buf7)
        del arg7_1
        del buf6
        buf11 = reinterpret_tensor(buf7, (1, 5, 32, 1536), (245760, 49152, 1536, 1), 0); del buf7  # reuse
        # Topologically Sorted Source Nodes: [context_tokens_1, projected], Original ATen: [aten._to_copy, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_4.run(buf11, arg8_1, arg9_1, arg10_1, 160, 1536, stream=stream0)
        del arg10_1
        del arg8_1
        del arg9_1
        buf12 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg11_1, (1536, 1536), (1, 1536), 0), out=buf12)
        del arg11_1
        buf14 = reinterpret_tensor(buf12, (5, 32, 1536), (49152, 1536, 1), 0); del buf12  # reuse
        # Topologically Sorted Source Nodes: [float_1, pow_1, mean, add, rsqrt, mul, to_2, mul_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf14, arg12_1, arg13_1, 160, 1536, stream=stream0)
        del arg12_1
        del arg13_1
        buf15 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg15_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg14_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf15)
        del arg14_1
        del arg15_1
        buf16 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg16_1, (1536, 1536), (1, 1536), 0), out=buf16)
        del arg16_1
        buf18 = reinterpret_tensor(buf16, (5, 32, 1536), (49152, 1536, 1), 0); del buf16  # reuse
        # Topologically Sorted Source Nodes: [float_2, pow_2, mean_1, add_1, rsqrt_1, mul_2, to_3, mul_3], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf18, arg17_1, arg18_1, 160, 1536, stream=stream0)
        del arg17_1
        del arg18_1
        buf19 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg20_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg19_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf19)
        del arg19_1
        del arg20_1
        buf20 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_8], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg21_1, (1536, 1536), (1, 1536), 0), out=buf20)
        del arg21_1
        buf22 = reinterpret_tensor(buf20, (5, 32, 1536), (49152, 1536, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [float_3, pow_3, mean_2, add_2, rsqrt_2, mul_4, to_4, mul_5], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf22, arg22_1, arg23_1, 160, 1536, stream=stream0)
        del arg22_1
        del arg23_1
        buf23 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg25_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg24_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf23)
        del arg24_1
        del arg25_1
        buf24 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg26_1, (1536, 1536), (1, 1536), 0), out=buf24)
        del arg26_1
        buf26 = reinterpret_tensor(buf24, (5, 32, 1536), (49152, 1536, 1), 0); del buf24  # reuse
        # Topologically Sorted Source Nodes: [float_4, pow_4, mean_3, add_3, rsqrt_3, mul_6, to_5, mul_7], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf26, arg27_1, arg28_1, 160, 1536, stream=stream0)
        del arg27_1
        del arg28_1
        buf27 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_11], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg30_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg29_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf27)
        del arg29_1
        del arg30_1
        buf28 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg31_1, (1536, 1536), (1, 1536), 0), out=buf28)
        del arg31_1
        buf30 = reinterpret_tensor(buf28, (5, 32, 1536), (49152, 1536, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [float_5, pow_5, mean_4, add_4, rsqrt_4, mul_8, to_6, mul_9], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf30, arg32_1, arg33_1, 160, 1536, stream=stream0)
        del arg32_1
        del arg33_1
        buf31 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_13], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg35_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg34_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf31)
        del arg34_1
        del arg35_1
        buf32 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_14], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg36_1, (1536, 1536), (1, 1536), 0), out=buf32)
        del arg36_1
        buf34 = reinterpret_tensor(buf32, (5, 32, 1536), (49152, 1536, 1), 0); del buf32  # reuse
        # Topologically Sorted Source Nodes: [float_6, pow_6, mean_5, add_5, rsqrt_5, mul_10, to_7, mul_11], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf34, arg37_1, arg38_1, 160, 1536, stream=stream0)
        del arg37_1
        del arg38_1
        buf35 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_15], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg40_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg39_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf35)
        del arg39_1
        del arg40_1
        buf36 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_16], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg41_1, (1536, 1536), (1, 1536), 0), out=buf36)
        del arg41_1
        buf38 = reinterpret_tensor(buf36, (5, 32, 1536), (49152, 1536, 1), 0); del buf36  # reuse
        # Topologically Sorted Source Nodes: [float_7, pow_7, mean_6, add_6, rsqrt_6, mul_12, to_8, mul_13], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf38, arg42_1, arg43_1, 160, 1536, stream=stream0)
        del arg42_1
        del arg43_1
        buf39 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_17], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg45_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg44_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf39)
        del arg44_1
        del arg45_1
        buf40 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_18], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg46_1, (1536, 1536), (1, 1536), 0), out=buf40)
        del arg46_1
        buf42 = reinterpret_tensor(buf40, (5, 32, 1536), (49152, 1536, 1), 0); del buf40  # reuse
        # Topologically Sorted Source Nodes: [float_8, pow_8, mean_7, add_7, rsqrt_7, mul_14, to_9, mul_15], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf42, arg47_1, arg48_1, 160, 1536, stream=stream0)
        del arg47_1
        del arg48_1
        buf43 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_19], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg50_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg49_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf43)
        del arg49_1
        del arg50_1
        buf44 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_20], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg51_1, (1536, 1536), (1, 1536), 0), out=buf44)
        del arg51_1
        buf46 = reinterpret_tensor(buf44, (5, 32, 1536), (49152, 1536, 1), 0); del buf44  # reuse
        # Topologically Sorted Source Nodes: [float_9, pow_9, mean_8, add_8, rsqrt_8, mul_16, to_10, mul_17], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf46, arg52_1, arg53_1, 160, 1536, stream=stream0)
        del arg52_1
        del arg53_1
        buf47 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_21], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg55_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg54_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf47)
        del arg54_1
        del arg55_1
        buf48 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_22], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg56_1, (1536, 1536), (1, 1536), 0), out=buf48)
        del arg56_1
        buf50 = reinterpret_tensor(buf48, (5, 32, 1536), (49152, 1536, 1), 0); del buf48  # reuse
        # Topologically Sorted Source Nodes: [float_10, pow_10, mean_9, add_9, rsqrt_9, mul_18, to_11, mul_19], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf50, arg57_1, arg58_1, 160, 1536, stream=stream0)
        del arg57_1
        del arg58_1
        buf51 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_23], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg60_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg59_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf51)
        del arg59_1
        del arg60_1
        buf52 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_24], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg61_1, (1536, 1536), (1, 1536), 0), out=buf52)
        del arg61_1
        buf54 = reinterpret_tensor(buf52, (5, 32, 1536), (49152, 1536, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [float_11, pow_11, mean_10, add_10, rsqrt_10, mul_20, to_12, mul_21], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf54, arg62_1, arg63_1, 160, 1536, stream=stream0)
        del arg62_1
        del arg63_1
        buf55 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_25], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg65_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg64_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf55)
        del arg64_1
        del arg65_1
        buf56 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_26], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg66_1, (1536, 1536), (1, 1536), 0), out=buf56)
        del arg66_1
        buf58 = reinterpret_tensor(buf56, (5, 32, 1536), (49152, 1536, 1), 0); del buf56  # reuse
        # Topologically Sorted Source Nodes: [float_12, pow_12, mean_11, add_11, rsqrt_11, mul_22, to_13, mul_23], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf58, arg67_1, arg68_1, 160, 1536, stream=stream0)
        del arg67_1
        del arg68_1
        buf59 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_27], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg70_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg69_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf59)
        del arg69_1
        del arg70_1
        buf60 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_28], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg71_1, (1536, 1536), (1, 1536), 0), out=buf60)
        del arg71_1
        buf62 = reinterpret_tensor(buf60, (5, 32, 1536), (49152, 1536, 1), 0); del buf60  # reuse
        # Topologically Sorted Source Nodes: [float_13, pow_13, mean_12, add_12, rsqrt_12, mul_24, to_14, mul_25], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf62, arg72_1, arg73_1, 160, 1536, stream=stream0)
        del arg72_1
        del arg73_1
        buf63 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_29], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg75_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg74_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf63)
        del arg74_1
        del arg75_1
        buf64 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_30], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg76_1, (1536, 1536), (1, 1536), 0), out=buf64)
        del arg76_1
        buf66 = reinterpret_tensor(buf64, (5, 32, 1536), (49152, 1536, 1), 0); del buf64  # reuse
        # Topologically Sorted Source Nodes: [float_14, pow_14, mean_13, add_13, rsqrt_13, mul_26, to_15, mul_27], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf66, arg77_1, arg78_1, 160, 1536, stream=stream0)
        del arg77_1
        del arg78_1
        buf67 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_31], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg80_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg79_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf67)
        del arg79_1
        del arg80_1
        buf68 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_32], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg81_1, (1536, 1536), (1, 1536), 0), out=buf68)
        del arg81_1
        buf70 = reinterpret_tensor(buf68, (5, 32, 1536), (49152, 1536, 1), 0); del buf68  # reuse
        # Topologically Sorted Source Nodes: [float_15, pow_15, mean_14, add_14, rsqrt_14, mul_28, to_16, mul_29], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf70, arg82_1, arg83_1, 160, 1536, stream=stream0)
        del arg82_1
        del arg83_1
        buf71 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_33], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg85_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg84_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf71)
        del arg84_1
        del arg85_1
        buf72 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_34], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg86_1, (1536, 1536), (1, 1536), 0), out=buf72)
        del arg86_1
        buf74 = reinterpret_tensor(buf72, (5, 32, 1536), (49152, 1536, 1), 0); del buf72  # reuse
        # Topologically Sorted Source Nodes: [float_16, pow_16, mean_15, add_15, rsqrt_15, mul_30, to_17, mul_31], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf74, arg87_1, arg88_1, 160, 1536, stream=stream0)
        del arg87_1
        del arg88_1
        buf75 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_35], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg90_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg89_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf75)
        del arg89_1
        del arg90_1
        buf76 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_36], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg91_1, (1536, 1536), (1, 1536), 0), out=buf76)
        del arg91_1
        buf78 = reinterpret_tensor(buf76, (5, 32, 1536), (49152, 1536, 1), 0); del buf76  # reuse
        # Topologically Sorted Source Nodes: [float_17, pow_17, mean_16, add_16, rsqrt_16, mul_32, to_18, mul_33], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf78, arg92_1, arg93_1, 160, 1536, stream=stream0)
        del arg92_1
        del arg93_1
        buf79 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_37], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg95_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg94_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf79)
        del arg94_1
        del arg95_1
        buf80 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_38], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg96_1, (1536, 1536), (1, 1536), 0), out=buf80)
        del arg96_1
        buf82 = reinterpret_tensor(buf80, (5, 32, 1536), (49152, 1536, 1), 0); del buf80  # reuse
        # Topologically Sorted Source Nodes: [float_18, pow_18, mean_17, add_17, rsqrt_17, mul_34, to_19, mul_35], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf82, arg97_1, arg98_1, 160, 1536, stream=stream0)
        del arg97_1
        del arg98_1
        buf83 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_39], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg100_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg99_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf83)
        del arg100_1
        del arg99_1
        buf84 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_40], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg101_1, (1536, 1536), (1, 1536), 0), out=buf84)
        del arg101_1
        buf86 = reinterpret_tensor(buf84, (5, 32, 1536), (49152, 1536, 1), 0); del buf84  # reuse
        # Topologically Sorted Source Nodes: [float_19, pow_19, mean_18, add_18, rsqrt_18, mul_36, to_20, mul_37], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf86, arg102_1, arg103_1, 160, 1536, stream=stream0)
        del arg102_1
        del arg103_1
        buf87 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_41], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg105_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg104_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf87)
        del arg104_1
        del arg105_1
        buf88 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_42], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg106_1, (1536, 1536), (1, 1536), 0), out=buf88)
        del arg106_1
        buf90 = reinterpret_tensor(buf88, (5, 32, 1536), (49152, 1536, 1), 0); del buf88  # reuse
        # Topologically Sorted Source Nodes: [float_20, pow_20, mean_19, add_19, rsqrt_19, mul_38, to_21, mul_39], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf90, arg107_1, arg108_1, 160, 1536, stream=stream0)
        del arg107_1
        del arg108_1
        buf91 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_43], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg110_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg109_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf91)
        del arg109_1
        del arg110_1
        buf92 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_44], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg111_1, (1536, 1536), (1, 1536), 0), out=buf92)
        del arg111_1
        buf94 = reinterpret_tensor(buf92, (5, 32, 1536), (49152, 1536, 1), 0); del buf92  # reuse
        # Topologically Sorted Source Nodes: [float_21, pow_21, mean_20, add_20, rsqrt_20, mul_40, to_22, mul_41], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf94, arg112_1, arg113_1, 160, 1536, stream=stream0)
        del arg112_1
        del arg113_1
        buf95 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_45], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg115_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg114_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf95)
        del arg114_1
        del arg115_1
        buf96 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_46], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg116_1, (1536, 1536), (1, 1536), 0), out=buf96)
        del arg116_1
        buf98 = reinterpret_tensor(buf96, (5, 32, 1536), (49152, 1536, 1), 0); del buf96  # reuse
        # Topologically Sorted Source Nodes: [float_22, pow_22, mean_21, add_21, rsqrt_21, mul_42, to_23, mul_43], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf98, arg117_1, arg118_1, 160, 1536, stream=stream0)
        del arg117_1
        del arg118_1
        buf99 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_47], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg120_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg119_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf99)
        del arg119_1
        del arg120_1
        buf100 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_48], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg121_1, (1536, 1536), (1, 1536), 0), out=buf100)
        del arg121_1
        buf102 = reinterpret_tensor(buf100, (5, 32, 1536), (49152, 1536, 1), 0); del buf100  # reuse
        # Topologically Sorted Source Nodes: [float_23, pow_23, mean_22, add_22, rsqrt_22, mul_44, to_24, mul_45], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf102, arg122_1, arg123_1, 160, 1536, stream=stream0)
        del arg122_1
        del arg123_1
        buf103 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_49], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg125_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg124_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf103)
        del arg124_1
        del arg125_1
        buf104 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_50], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg126_1, (1536, 1536), (1, 1536), 0), out=buf104)
        del arg126_1
        buf106 = reinterpret_tensor(buf104, (5, 32, 1536), (49152, 1536, 1), 0); del buf104  # reuse
        # Topologically Sorted Source Nodes: [float_24, pow_24, mean_23, add_23, rsqrt_23, mul_46, to_25, mul_47], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf106, arg127_1, arg128_1, 160, 1536, stream=stream0)
        del arg127_1
        del arg128_1
        buf107 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_51], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg130_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg129_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf107)
        del arg129_1
        del arg130_1
        buf108 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_52], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg131_1, (1536, 1536), (1, 1536), 0), out=buf108)
        del arg131_1
        buf110 = reinterpret_tensor(buf108, (5, 32, 1536), (49152, 1536, 1), 0); del buf108  # reuse
        # Topologically Sorted Source Nodes: [float_25, pow_25, mean_24, add_24, rsqrt_24, mul_48, to_26, mul_49], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf110, arg132_1, arg133_1, 160, 1536, stream=stream0)
        del arg132_1
        del arg133_1
        buf111 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_53], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg135_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg134_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf111)
        del arg134_1
        del arg135_1
        buf112 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_54], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg136_1, (1536, 1536), (1, 1536), 0), out=buf112)
        del arg136_1
        buf114 = reinterpret_tensor(buf112, (5, 32, 1536), (49152, 1536, 1), 0); del buf112  # reuse
        # Topologically Sorted Source Nodes: [float_26, pow_26, mean_25, add_25, rsqrt_25, mul_50, to_27, mul_51], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf114, arg137_1, arg138_1, 160, 1536, stream=stream0)
        del arg137_1
        del arg138_1
        buf115 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_55], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg140_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg139_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf115)
        del arg139_1
        del arg140_1
        buf116 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_56], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg141_1, (1536, 1536), (1, 1536), 0), out=buf116)
        del arg141_1
        buf118 = reinterpret_tensor(buf116, (5, 32, 1536), (49152, 1536, 1), 0); del buf116  # reuse
        # Topologically Sorted Source Nodes: [float_27, pow_27, mean_26, add_26, rsqrt_26, mul_52, to_28, mul_53], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf118, arg142_1, arg143_1, 160, 1536, stream=stream0)
        del arg142_1
        del arg143_1
        buf119 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_57], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg145_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg144_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf119)
        del arg144_1
        del arg145_1
        buf120 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_58], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg146_1, (1536, 1536), (1, 1536), 0), out=buf120)
        del arg146_1
        buf122 = reinterpret_tensor(buf120, (5, 32, 1536), (49152, 1536, 1), 0); del buf120  # reuse
        # Topologically Sorted Source Nodes: [float_28, pow_28, mean_27, add_27, rsqrt_27, mul_54, to_29, mul_55], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf122, arg147_1, arg148_1, 160, 1536, stream=stream0)
        del arg147_1
        del arg148_1
        buf123 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_59], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg150_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg149_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf123)
        del arg149_1
        del arg150_1
        buf124 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_60], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg151_1, (1536, 1536), (1, 1536), 0), out=buf124)
        del arg151_1
        buf126 = reinterpret_tensor(buf124, (5, 32, 1536), (49152, 1536, 1), 0); del buf124  # reuse
        # Topologically Sorted Source Nodes: [float_29, pow_29, mean_28, add_28, rsqrt_28, mul_56, to_30, mul_57], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf126, arg152_1, arg153_1, 160, 1536, stream=stream0)
        del arg152_1
        del arg153_1
        buf127 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_61], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg155_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg154_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf127)
        del arg154_1
        del arg155_1
        buf128 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_62], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg156_1, (1536, 1536), (1, 1536), 0), out=buf128)
        del arg156_1
        buf130 = reinterpret_tensor(buf128, (5, 32, 1536), (49152, 1536, 1), 0); del buf128  # reuse
        # Topologically Sorted Source Nodes: [float_30, pow_30, mean_29, add_29, rsqrt_29, mul_58, to_31, mul_59], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_5.run(buf130, arg157_1, arg158_1, 160, 1536, stream=stream0)
        del arg157_1
        del arg158_1
        buf131 = empty_strided_cuda((160, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_63], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg160_1, reinterpret_tensor(buf11, (160, 1536), (1536, 1), 0), reinterpret_tensor(arg159_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf131)
        del arg159_1
        del arg160_1
    return (buf11, buf14, reinterpret_tensor(buf15, (5, 32, 1536), (49152, 1536, 1), 0), buf18, reinterpret_tensor(buf19, (5, 32, 1536), (49152, 1536, 1), 0), buf22, reinterpret_tensor(buf23, (5, 32, 1536), (49152, 1536, 1), 0), buf26, reinterpret_tensor(buf27, (5, 32, 1536), (49152, 1536, 1), 0), buf30, reinterpret_tensor(buf31, (5, 32, 1536), (49152, 1536, 1), 0), buf34, reinterpret_tensor(buf35, (5, 32, 1536), (49152, 1536, 1), 0), buf38, reinterpret_tensor(buf39, (5, 32, 1536), (49152, 1536, 1), 0), buf42, reinterpret_tensor(buf43, (5, 32, 1536), (49152, 1536, 1), 0), buf46, reinterpret_tensor(buf47, (5, 32, 1536), (49152, 1536, 1), 0), buf50, reinterpret_tensor(buf51, (5, 32, 1536), (49152, 1536, 1), 0), buf54, reinterpret_tensor(buf55, (5, 32, 1536), (49152, 1536, 1), 0), buf58, reinterpret_tensor(buf59, (5, 32, 1536), (49152, 1536, 1), 0), buf62, reinterpret_tensor(buf63, (5, 32, 1536), (49152, 1536, 1), 0), buf66, reinterpret_tensor(buf67, (5, 32, 1536), (49152, 1536, 1), 0), buf70, reinterpret_tensor(buf71, (5, 32, 1536), (49152, 1536, 1), 0), buf74, reinterpret_tensor(buf75, (5, 32, 1536), (49152, 1536, 1), 0), buf78, reinterpret_tensor(buf79, (5, 32, 1536), (49152, 1536, 1), 0), buf82, reinterpret_tensor(buf83, (5, 32, 1536), (49152, 1536, 1), 0), buf86, reinterpret_tensor(buf87, (5, 32, 1536), (49152, 1536, 1), 0), buf90, reinterpret_tensor(buf91, (5, 32, 1536), (49152, 1536, 1), 0), buf94, reinterpret_tensor(buf95, (5, 32, 1536), (49152, 1536, 1), 0), buf98, reinterpret_tensor(buf99, (5, 32, 1536), (49152, 1536, 1), 0), buf102, reinterpret_tensor(buf103, (5, 32, 1536), (49152, 1536, 1), 0), buf106, reinterpret_tensor(buf107, (5, 32, 1536), (49152, 1536, 1), 0), buf110, reinterpret_tensor(buf111, (5, 32, 1536), (49152, 1536, 1), 0), buf114, reinterpret_tensor(buf115, (5, 32, 1536), (49152, 1536, 1), 0), buf118, reinterpret_tensor(buf119, (5, 32, 1536), (49152, 1536, 1), 0), buf122, reinterpret_tensor(buf123, (5, 32, 1536), (49152, 1536, 1), 0), buf126, reinterpret_tensor(buf127, (5, 32, 1536), (49152, 1536, 1), 0), buf130, reinterpret_tensor(buf131, (5, 32, 1536), (49152, 1536, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 33, 5, 12, 768), (1520640, 46080, 9216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((512, 46080), (46080, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((512, 110592), (110592, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((512, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((49152, 512), (512, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((49152, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg98_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg99_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg100_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg101_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg102_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg103_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg104_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg105_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg106_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg107_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg108_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg109_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg110_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg111_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg112_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg113_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg114_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg115_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg116_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg117_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg118_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg119_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg120_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg121_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg122_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg123_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg124_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg125_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg126_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg127_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg128_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg129_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg130_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg131_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg132_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg133_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg134_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg135_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg136_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg137_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg138_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg139_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg140_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg141_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg142_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg143_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg144_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg145_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg146_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg147_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg148_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg149_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg150_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg151_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg152_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg153_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg154_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg155_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg156_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg157_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg158_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg159_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg160_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
