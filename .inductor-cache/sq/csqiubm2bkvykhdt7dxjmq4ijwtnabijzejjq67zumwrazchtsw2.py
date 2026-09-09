# AOT ID: ['4_inference']
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/6e/c6ee57qra3udjqf53cwpkcfafbyfrkmoltce5iaiw2ywnelrpeoy.py
# Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add => add
# Graph fragment:
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg0_1, %arg1_1), kwargs = {})
triton_poi_fused_add_0 = async_compile.triton('triton_poi_fused_add_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/wf/cwfh4zwfta6fn3spjyazsjqofh4nrr6hknxydfhv752s4h4oy4hb.py
# Topologically Sorted Source Nodes: [layer_norm, add_1, mul, add_2], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_2
#   add_2 => add_3
#   layer_norm => add_1, convert_element_type, convert_element_type_1, mul, rsqrt, sub, var_mean
#   mul => mul_1
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%arg2_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_7), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_6, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1, 1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %add_2), kwargs = {})
#   %add_3 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %getitem), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_1 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 2, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_1(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1950
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp3_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp3_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp3_mean_next, tmp3_m2_next, tmp3_weight_next = triton_helpers.welford_reduce(
            tmp2, tmp3_mean, tmp3_m2, tmp3_weight, roffset == 0
        )
        tmp3_mean = tl.where(r0_mask & xmask, tmp3_mean_next, tmp3_mean)
        tmp3_m2 = tl.where(r0_mask & xmask, tmp3_m2_next, tmp3_m2)
        tmp3_weight = tl.where(r0_mask & xmask, tmp3_weight_next, tmp3_weight)
    tmp6, tmp7, tmp8 = triton_helpers.welford(tmp3_mean, tmp3_m2, tmp3_weight, 1)
    tmp3 = tmp6[:, None]
    tmp4 = tmp7[:, None]
    tmp5 = tmp8[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp9 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr1 + (1536 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp23 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp3
        tmp12 = 1536.0
        tmp13 = (tmp4 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = 1.0
        tmp21 = tmp19 + tmp20
        tmp22 = tmp18 * tmp21
        tmp24 = tmp22 + tmp23
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp24, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/he/cheqoy7hheex2g47ir5jzc7skalzbjeojh2bcwjsn6yfw5qoaw2n.py
# Topologically Sorted Source Nodes: [float_1, pow_1, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   float_1 => convert_element_type_5
#   mean => mean
#   pow_1 => pow_1
# Graph fragment:
#   %convert_element_type_5 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_5, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [-1], True), kwargs = {})
triton_red_fused__to_copy_mean_pow_2 = async_compile.triton('triton_red_fused__to_copy_mean_pow_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_mean_pow_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_mean_pow_2(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1950
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/az/cazy2ejqrky2tyzr7axzlbsmfh7qgj3bver6ioc7y5gxdwr4xk2b.py
# Topologically Sorted Source Nodes: [mul_5, mul_6, sub, mul_7, mul_8, add_5, mul_9, mul_10, sub_1, mul_11, mul_12, add_6], Original ATen: [aten.mul, aten.sub, aten.add]
# Source node to ATen node mapping:
#   add_5 => add_6
#   add_6 => add_7
#   mul_10 => mul_11
#   mul_11 => mul_12
#   mul_12 => mul_13
#   mul_5 => mul_6
#   mul_6 => mul_7
#   mul_7 => mul_8
#   mul_8 => mul_9
#   mul_9 => mul_10
#   sub => sub_1
#   sub_1 => sub_2
# Graph fragment:
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %getitem_10), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_9, %getitem_11), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_6, %mul_7), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_8, %getitem_11), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_9, %getitem_10), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_8, %mul_9), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_12, %getitem_14), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_13, %getitem_15), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_10, %mul_11), kwargs = {})
#   %mul_12 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_12, %getitem_15), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%getitem_13, %getitem_14), kwargs = {})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_12, %mul_13), kwargs = {})
triton_poi_fused_add_mul_sub_3 = async_compile.triton('triton_poi_fused_add_mul_sub_3', '''
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
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/6h/c6hrzwjcvflnfa3tulfb5apx3jgl24oqjtegjrlt64fz3e4geymi.py
# Topologically Sorted Source Nodes: [to_3, to_4, _flash_attn_forward], Original ATen: [aten._to_copy, flash_attn._flash_attn_forward]
# Source node to ATen node mapping:
#   _flash_attn_forward => _flash_attn_forward
#   to_3 => convert_element_type_16
#   to_4 => convert_element_type_18
# Graph fragment:
#   %convert_element_type_16 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_9, torch.bfloat16), kwargs = {})
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_12, torch.bfloat16), kwargs = {})
#   %_flash_attn_forward : [num_users=1] = call_function[target=torch.ops.flash_attn._flash_attn_forward.default](args = (%convert_element_type_16, %convert_element_type_18, %view_16, 0.0, 0.08838834764831845, False, -1, -1, 0.0, None, False), kwargs = {})
triton_poi_fused__flash_attn_forward__to_copy_4 = async_compile.triton('triton_poi_fused__flash_attn_forward__to_copy_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__flash_attn_forward__to_copy_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__flash_attn_forward__to_copy_4(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2995200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = (x2 % 2)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 // 2), xmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 2, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x2 // 2), xmask & tmp6, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp11 = tmp10.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp11, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/jd/cjd3x5sqr5xmcz4ny2b4y2st4aewncwxb3ejh2q72jgcljvmx43d.py
# Topologically Sorted Source Nodes: [mul_13, x_1, layer_norm_1], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   layer_norm_1 => add_10, add_9, convert_element_type_22, convert_element_type_23, mul_15, mul_16, rsqrt_3, sub_3, var_mean_1
#   mul_13 => mul_14
#   x_1 => add_8
# Graph fragment:
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %getitem_2), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mul_14), kwargs = {})
#   %convert_element_type_22 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_22, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_3 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_22, %getitem_21), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_20, 1e-06), kwargs = {})
#   %rsqrt_3 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_9,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_3, %rsqrt_3), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_15, %arg14_1), kwargs = {})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_16, %arg15_1), kwargs = {})
#   %convert_element_type_23 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_10, torch.bfloat16), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_5 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 10, 'num_reduction': 2, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1950
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp9_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp9_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (3072 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp9_mean_next, tmp9_m2_next, tmp9_weight_next = triton_helpers.welford_reduce(
            tmp8, tmp9_mean, tmp9_m2, tmp9_weight, roffset == 0
        )
        tmp9_mean = tl.where(r0_mask & xmask, tmp9_mean_next, tmp9_mean)
        tmp9_m2 = tl.where(r0_mask & xmask, tmp9_m2_next, tmp9_m2)
        tmp9_weight = tl.where(r0_mask & xmask, tmp9_weight_next, tmp9_weight)
    tmp12, tmp13, tmp14 = triton_helpers.welford(tmp9_mean, tmp9_m2, tmp9_weight, 1)
    tmp9 = tmp12[:, None]
    tmp10 = tmp13[:, None]
    tmp11 = tmp14[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp15 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp19 = tl.load(in_ptr3 + (3072 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 + tmp17
        tmp20 = tmp18 * tmp19
        tmp21 = tmp15 + tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp9
        tmp24 = 1536.0
        tmp25 = (tmp10 / tmp24)
        tmp26 = 1e-06
        tmp27 = tmp25 + tmp26
        tmp28 = libdevice.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 * tmp31
        tmp34 = tmp33.to(tl.float32)
        tmp35 = tmp32 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp36, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ai/cai2t75qpb7hlx5w5udmfpo3r7wb6p2fkxgndjnte4t6acjabsp6.py
# Topologically Sorted Source Nodes: [float_5, pow_3, mean_2, _flash_attn_forward_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, flash_attn._flash_attn_forward]
# Source node to ATen node mapping:
#   _flash_attn_forward_1 => _flash_attn_forward_1
#   float_5 => convert_element_type_27
#   mean_2 => mean_2
#   pow_3 => pow_3
# Graph fragment:
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_24, torch.float32), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_27, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [-1], True), kwargs = {})
#   %_flash_attn_forward_1 : [num_users=1] = call_function[target=torch.ops.flash_attn._flash_attn_forward.default](args = (%view_25, %view_26, %view_27, 0.0, 0.08838834764831845, False, -1, -1, 0.0, None, False), kwargs = {})
triton_red_fused__flash_attn_forward__to_copy_mean_pow_6 = async_compile.triton('triton_red_fused__flash_attn_forward__to_copy_mean_pow_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__flash_attn_forward__to_copy_mean_pow_6', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__flash_attn_forward__to_copy_mean_pow_6(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1950
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/xr/cxr4rj6vq4skffeyxrr6hy6cntvnhbm6r4eqt3mjoqoydlk5aqlk.py
# Topologically Sorted Source Nodes: [mul_13, x_1, x_4, layer_norm_2, add_10, mul_16, add_11], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add_10 => add_14
#   add_11 => add_15
#   layer_norm_2 => add_13, convert_element_type_32, convert_element_type_33, mul_19, rsqrt_5, sub_4, var_mean_2
#   mul_13 => mul_14
#   mul_16 => mul_20
#   x_1 => add_8
#   x_4 => add_12
# Graph fragment:
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%view_19, %getitem_2), kwargs = {})
#   %add_8 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%arg2_1, %mul_14), kwargs = {})
#   %add_12 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %view_32), kwargs = {})
#   %convert_element_type_32 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_12, torch.float32), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_32, [2]), kwargs = {correction: 0, keepdim: True})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_32, %getitem_27), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_26, 1e-06), kwargs = {})
#   %rsqrt_5 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_13,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_5), kwargs = {})
#   %convert_element_type_33 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_19, torch.bfloat16), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_4, 1), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_33, %add_14), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_20, %getitem_3), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_7 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 2048, 'r0_': 2048},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1950
    r0_numel = 1536
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (3072 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 * tmp4
        tmp6 = tmp0 + tmp5
        tmp9 = tmp7 + tmp8
        tmp10 = tmp6 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(r0_mask & xmask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask & xmask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask & xmask, tmp13_weight_next, tmp13_weight)
        tl.store(in_out_ptr0 + (r0_1 + 1536*x0), tmp10, r0_mask & xmask)
    tmp16, tmp17, tmp18 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp16[:, None]
    tmp14 = tmp17[:, None]
    tmp15 = tmp18[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp19 = tl.load(in_out_ptr0 + (r0_1 + 1536*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp29 = tl.load(in_ptr2 + (6144 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp33 = tl.load(in_ptr2 + (4608 + r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 - tmp13
        tmp22 = 1536.0
        tmp23 = (tmp14 / tmp22)
        tmp24 = 1e-06
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp30 = 1.0
        tmp31 = tmp29 + tmp30
        tmp32 = tmp28 * tmp31
        tmp34 = tmp32 + tmp33
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp34, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 6, 1536), (9216, 1536, 1))
    assert_size_stride(arg1_1, (1, 6, 1536), (9216, 1536, 1))
    assert_size_stride(arg2_1, (1, 1950, 1536), (2995200, 1536, 1))
    assert_size_stride(arg3_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg4_1, (1536, ), (1, ))
    assert_size_stride(arg5_1, (1536, ), (1, ))
    assert_size_stride(arg6_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg7_1, (1536, ), (1, ))
    assert_size_stride(arg8_1, (1536, ), (1, ))
    assert_size_stride(arg9_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg10_1, (1536, ), (1, ))
    assert_size_stride(arg11_1, (1950, 1, 64, 2), (128, 128, 2, 1))
    assert_size_stride(arg12_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg13_1, (1536, ), (1, ))
    assert_size_stride(arg14_1, (1536, ), (1, ))
    assert_size_stride(arg15_1, (1536, ), (1, ))
    assert_size_stride(arg16_1, (1, 5, 32, 1536), (245760, 49152, 1536, 1))
    assert_size_stride(arg17_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg18_1, (1536, ), (1, ))
    assert_size_stride(arg19_1, (1536, ), (1, ))
    assert_size_stride(arg20_1, (5, 32, 1536), (49152, 1536, 1))
    assert_size_stride(arg21_1, (5, 32, 1536), (49152, 1536, 1))
    assert_size_stride(arg22_1, (1536, 1536), (1536, 1))
    assert_size_stride(arg23_1, (1536, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 6, 1536), (9216, 1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [add], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_0.run(arg0_1, arg1_1, buf0, 9216, stream=stream0)
        del arg0_1
        del arg1_1
        buf4 = empty_strided_cuda((1, 1950, 1536), (2995200, 1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer_norm, add_1, mul, add_2], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_native_layer_norm_1.run(arg2_1, buf0, buf4, 1950, 1536, stream=stream0)
        buf5 = empty_strided_cuda((1950, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf4, (1950, 1536), (1536, 1), 0), reinterpret_tensor(arg3_1, (1536, 1536), (1, 1536), 0), out=buf5)
        del arg3_1
        buf6 = empty_strided_cuda((1, 1950, 1), (1952, 1, 1952), torch.float32)
        # Topologically Sorted Source Nodes: [float_1, pow_1, mean], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_mean_pow_2.run(buf5, arg4_1, buf6, 1950, 1536, stream=stream0)
        buf7 = empty_strided_cuda((1950, 1536), (1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf4, (1950, 1536), (1536, 1), 0), reinterpret_tensor(arg6_1, (1536, 1536), (1, 1536), 0), out=buf7)
        del arg6_1
        buf8 = empty_strided_cuda((1, 1950, 1), (1952, 1, 1952), torch.float32)
        # Topologically Sorted Source Nodes: [float_2, pow_2, mean_1], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_mean_pow_2.run(buf7, arg7_1, buf8, 1950, 1536, stream=stream0)
        buf9 = empty_strided_cuda((1, 1950, 12, 64), (1497600, 768, 64, 1), torch.float32)
        buf10 = empty_strided_cuda((1, 1950, 12, 64), (1497600, 768, 64, 1), torch.float32)
        buf11 = empty_strided_cuda((1, 1950, 12, 64), (1497600, 768, 64, 1), torch.float32)
        buf12 = empty_strided_cuda((1, 1950, 12, 64), (1497600, 768, 64, 1), torch.float32)
        # Topologically Sorted Source Nodes: [mul_5, mul_6, sub, mul_7, mul_8, add_5, mul_9, mul_10, sub_1, mul_11, mul_12, add_6], Original ATen: [aten.mul, aten.sub, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_mul_sub_3.run(buf5, arg4_1, buf6, arg5_1, arg11_1, buf7, arg7_1, buf8, arg8_1, buf9, buf10, buf11, buf12, 1497600, stream=stream0)
        del arg11_1
        del arg4_1
        del arg5_1
        del arg7_1
        del arg8_1
        del buf6
        del buf8
        buf13 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [v], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg10_1, reinterpret_tensor(buf4, (1950, 1536), (1536, 1), 0), reinterpret_tensor(arg9_1, (1536, 1536), (1, 1536), 0), alpha=1, beta=1, out=buf13)
        del arg10_1
        del arg9_1
        buf14 = reinterpret_tensor(buf4, (1, 1950, 12, 128), (2995200, 1536, 128, 1), 0); del buf4  # reuse
        # Topologically Sorted Source Nodes: [to_3, to_4, _flash_attn_forward], Original ATen: [aten._to_copy, flash_attn._flash_attn_forward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__flash_attn_forward__to_copy_4.run(buf9, buf10, buf14, 2995200, stream=stream0)
        del buf10
        del buf9
        buf15 = reinterpret_tensor(buf5, (1, 1950, 12, 128), (2995200, 1536, 128, 1), 0); del buf5  # reuse
        # Topologically Sorted Source Nodes: [to_3, to_4, _flash_attn_forward], Original ATen: [aten._to_copy, flash_attn._flash_attn_forward]
        stream0 = get_raw_stream(0)
        triton_poi_fused__flash_attn_forward__to_copy_4.run(buf11, buf12, buf15, 2995200, stream=stream0)
        del buf11
        del buf12
        # Topologically Sorted Source Nodes: [to_3, to_4, _flash_attn_forward], Original ATen: [aten._to_copy, flash_attn._flash_attn_forward]
        buf16 = torch.ops.flash_attn._flash_attn_forward.default(buf14, buf15, reinterpret_tensor(buf13, (1, 1950, 12, 128), (2995200, 1536, 128, 1), 0), 0.0, 0.08838834764831845, False, -1, -1, 0.0, None, False)
        del buf13
        buf17 = buf16[0]
        assert_size_stride(buf17, (1, 1950, 12, 128), (2995200, 1536, 128, 1))
        del buf16
        buf21 = reinterpret_tensor(buf15, (1950, 1536), (1536, 1), 0); del buf15  # reuse
        # Topologically Sorted Source Nodes: [y], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf17, (1950, 1536), (1536, 1), 0), reinterpret_tensor(arg12_1, (1536, 1536), (1, 1536), 0), out=buf21)
        del arg12_1
        buf25 = reinterpret_tensor(buf17, (1, 1950, 1536), (2995200, 1536, 1), 0); del buf17  # reuse
        # Topologically Sorted Source Nodes: [mul_13, x_1, layer_norm_1], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_native_layer_norm_5.run(arg2_1, buf21, arg13_1, buf0, arg14_1, arg15_1, buf25, 1950, 1536, stream=stream0)
        del arg14_1
        del arg15_1
        buf26 = reinterpret_tensor(buf14, (1950, 1536), (1536, 1), 0); del buf14  # reuse
        # Topologically Sorted Source Nodes: [linear_4], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf25, (1950, 1536), (1536, 1), 0), reinterpret_tensor(arg17_1, (1536, 1536), (1, 1536), 0), out=buf26)
        del arg17_1
        del buf25
        buf28 = reinterpret_tensor(buf26, (5, 390, 12, 128), (599040, 1536, 128, 1), 0); del buf26  # reuse
        # Topologically Sorted Source Nodes: [float_5, pow_3, mean_2, _flash_attn_forward_1], Original ATen: [aten._to_copy, aten.pow, aten.mean, flash_attn._flash_attn_forward]
        stream0 = get_raw_stream(0)
        triton_red_fused__flash_attn_forward__to_copy_mean_pow_6.run(buf28, arg18_1, arg19_1, 1950, 1536, stream=stream0)
        del arg18_1
        del arg19_1
        # Topologically Sorted Source Nodes: [_flash_attn_forward_1], Original ATen: [flash_attn._flash_attn_forward]
        buf29 = torch.ops.flash_attn._flash_attn_forward.default(buf28, reinterpret_tensor(arg20_1, (5, 32, 12, 128), (49152, 1536, 128, 1), 0), reinterpret_tensor(arg21_1, (5, 32, 12, 128), (49152, 1536, 128, 1), 0), 0.0, 0.08838834764831845, False, -1, -1, 0.0, None, False)
        del arg20_1
        del arg21_1
        buf30 = buf29[0]
        assert_size_stride(buf30, (5, 390, 12, 128), (599040, 1536, 128, 1))
        del buf29
        buf34 = reinterpret_tensor(buf28, (1950, 1536), (1536, 1), 0); del buf28  # reuse
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf30, (1950, 1536), (1536, 1), 0), reinterpret_tensor(arg22_1, (1536, 1536), (1, 1536), 0), out=buf34)
        del arg22_1
        buf35 = reinterpret_tensor(buf21, (1, 1950, 1536), (2995200, 1536, 1), 0); del buf21  # reuse
        buf39 = reinterpret_tensor(buf30, (1, 1950, 1536), (2995200, 1536, 1), 0); del buf30  # reuse
        # Topologically Sorted Source Nodes: [mul_13, x_1, x_4, layer_norm_2, add_10, mul_16, add_11], Original ATen: [aten.mul, aten.add, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_native_layer_norm_7.run(buf35, arg2_1, arg13_1, buf0, buf34, arg23_1, buf39, 1950, 1536, stream=stream0)
        del arg13_1
        del arg23_1
        del arg2_1
        del buf34
    return (buf39, buf35, reinterpret_tensor(buf0, (1, 1, 1536), (9216, 1536, 1), 0), reinterpret_tensor(buf0, (1, 1, 1536), (9216, 1536, 1), 1536), reinterpret_tensor(buf0, (1, 1, 1536), (9216, 1536, 1), 3072), reinterpret_tensor(buf0, (1, 1, 1536), (9216, 1536, 1), 4608), reinterpret_tensor(buf0, (1, 1, 1536), (9216, 1536, 1), 6144), reinterpret_tensor(buf0, (1, 1, 1536), (9216, 1536, 1), 7680), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 6, 1536), (9216, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 6, 1536), (9216, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 1950, 1536), (2995200, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((1950, 1, 64, 2), (128, 128, 2, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((1, 5, 32, 1536), (245760, 49152, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((5, 32, 1536), (49152, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((5, 32, 1536), (49152, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((1536, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
