# AOT ID: ['0_inference']
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/43/c43hgscarzg5qfdehjspazwvlnlek26r3nflzsz4mp2rfvk34ym4.py
# Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_1 => cat
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
triton_poi_fused_cat_0 = async_compile.triton('triton_poi_fused_cat_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_0(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 41932800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 24960) % 35)
    x0 = (xindex % 120)
    x1 = ((xindex // 120) % 208)
    x3 = xindex // 873600
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x0 + 480*((x3 % 4)) + 1920*x1 + 13178880*(x3 // 16) + (((x3 // 4) % 4))), xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 35, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (4*x0 + 480*((x3 % 4)) + 1920*x1 + 399360*((-2) + x2) + 13178880*(x3 // 16) + (((x3 // 4) % 4))), xmask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ka/cka4kkqgcxxmdjbkl6zsuov343v2o5mdvlvwy6h4jdoz3bqf5ty6.py
# Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean => mean
#   pow_1 => pow_1
#   x_1 => cat
#   x_2 => convolution
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg1_1, %arg2_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
triton_red_fused_cat_convolution_mean_pow_1 = async_compile.triton('triton_red_fused_cat_convolution_mean_pow_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_pow_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_pow_1(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 823680
    r0_numel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + 823680*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/cm/ccmnvsde4m6zmgeork24s4qfj55hvys6yumszw7khloi5dqutkq7.py
# Topologically Sorted Source Nodes: [first_frame_pad_1], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_1 => repeat_1
# Graph fragment:
#   %repeat_1 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_8, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_2 = async_compile.triton('triton_poi_fused_repeat_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6389760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24960)
    x2 = xindex // 49920
    x3 = (xindex % 49920)
    tmp0 = tl.load(in_ptr0 + (x0 + 823680*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 128.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 873600*x2), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ns/cnsnmaivcgss2ahp6vl5426gp7trcapfvqvx72s54gdkxxp52tny.py
# Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean, add, sqrt, hidden_states, hidden_states_1], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add => add
#   hidden_states => div
#   hidden_states_1 => convert_element_type, convert_element_type_1, mul, sigmoid
#   mean => mean
#   pow_1 => pow_1
#   sqrt => sqrt
#   x_1 => cat
#   x_2 => convolution
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg1_1, %arg2_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-08), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, %sqrt), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %sigmoid), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 105431040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 823680
    x0 = (xindex % 823680)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 128.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 873600*x1), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/do/cdomhoqw4brhtzbzckmkxk5jmdqw72gi4bkcdfmuol22gqhpxnj7.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   output_tensor => add_2
#   pow_3 => pow_3
#   x_1 => cat
#   x_2 => convolution
#   x_6 => convolution_2
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg1_1, %arg2_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg5_1, %arg6_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [1], True), kwargs = {})
triton_red_fused_add_cat_convolution_mean_pow_4 = async_compile.triton('triton_red_fused_add_cat_convolution_mean_pow_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_convolution_mean_pow_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_convolution_mean_pow_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 823680
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 823680*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 823680*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/t2/ct25zh6vad2fvslk2zpjesykbsituyp32dfdykrov7ssyiiemhhc.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2, add_3, sqrt_2, hidden_states_5, hidden_states_6], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_3 => add_3
#   hidden_states_5 => div_2
#   hidden_states_6 => convert_element_type_4
#   mean_2 => mean_2
#   output_tensor => add_2
#   pow_3 => pow_3
#   sqrt_2 => sqrt_2
#   x_1 => cat
#   x_2 => convolution
#   x_6 => convolution_2
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg1_1, %arg2_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg5_1, %arg6_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [1], True), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-08), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_3,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_2, %sqrt_2), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 105431040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 823680
    x0 = (xindex % 823680)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = 128.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1e-08
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/5k/c5kkugztnl4troti52oyr6wxys6hlfzycoze2w32kt6tlrq5xjdr.py
# Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_7 => cat_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_3, %convert_element_type_5], 2), kwargs = {})
triton_poi_fused_cat_6 = async_compile.triton('triton_poi_fused_cat_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 111820800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 24960) % 35)
    x0 = (xindex % 24960)
    x2 = xindex // 873600
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 823680*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 35, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 24960*((-2) + x1) + 823680*x2), tmp11, other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/lr/clrbg66tjcyz6movgbhkr5yi4n63izvojyxyrvi63fcgvldldy3r.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, x_10, output_tensor_1], Original ATen: [aten.cat, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor => add_2
#   output_tensor_1 => add_5
#   x_1 => cat
#   x_10 => convolution_4
#   x_2 => convolution
#   x_6 => convolution_2
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg1_1, %arg2_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg5_1, %arg6_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_2 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_4, %arg9_1, %arg10_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2, %convolution_4), kwargs = {})
triton_poi_fused_add_cat_convolution_7 = async_compile.triton('triton_poi_fused_add_cat_convolution_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 105431040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 823680
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x2), None).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/oq/coqehjepuxtyxfwj2nwmikeazv5mdnoxaijizjypyvblutd3xtdo.py
# Topologically Sorted Source Nodes: [pow_5, mean_4], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_4 => mean_4
#   pow_5 => pow_5
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [1], True), kwargs = {})
triton_red_fused_mean_pow_8 = async_compile.triton('triton_red_fused_mean_pow_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_8(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 823680
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 823680*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/fr/cfruu5cksoglpjwghc3kdtheqrkosw6ndz64g2qazmvxjecbviih.py
# Topologically Sorted Source Nodes: [first_frame_pad_5], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_5 => repeat_5
# Graph fragment:
#   %repeat_5 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_28, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_9 = async_compile.triton('triton_poi_fused_repeat_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_9(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6389760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24960)
    x2 = xindex // 49920
    x3 = (xindex % 49920)
    tmp0 = tl.load(in_ptr0 + (x0 + 823680*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = 128.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 873600*x2), tmp12, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/xv/cxv3izqmru4w5u7etbifxo5emuzyxq5ba2dzvwgfwsela76xp4i3.py
# Topologically Sorted Source Nodes: [pow_5, mean_4, add_6, sqrt_4, hidden_states_10, hidden_states_11], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_6 => add_6
#   hidden_states_10 => div_4
#   hidden_states_11 => convert_element_type_8, convert_element_type_9, mul_4, sigmoid_4
#   mean_4 => mean_4
#   pow_5 => pow_5
#   sqrt_4 => sqrt_4
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_5, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [1], True), kwargs = {})
#   %add_6 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-08), kwargs = {})
#   %sqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_6,), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_5, %sqrt_4), kwargs = {})
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.float32), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_8,), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %sigmoid_4), kwargs = {})
#   %convert_element_type_9 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_4, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_10 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_10(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 105431040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 823680)
    x1 = xindex // 823680
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = 128.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 873600*x1), tmp12, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/hp/chpaklhvyqhizrycqyogcd7pirwmimjvbkcl264zu6nftupbdwnb.py
# Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_6 => mean_6
#   output_tensor_2 => add_8
#   pow_7 => pow_7
#   x_14 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_6, %arg13_1, %arg14_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %convolution_6), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_11 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 1048576, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 823680
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 823680*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 823680*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/lb/clbtoo7solqjbgtv5bpvuoq77hjbejgma4t3qiqunt3f4obsrdvx.py
# Topologically Sorted Source Nodes: [first_frame_pad_7], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_7 => repeat_7
# Graph fragment:
#   %repeat_7 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_38, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_12 = async_compile.triton('triton_poi_fused_repeat_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6389760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24960)
    x2 = xindex // 49920
    x3 = (xindex % 49920)
    tmp0 = tl.load(in_ptr0 + (x0 + 823680*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 823680*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 128.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 873600*x2), tmp16, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/vr/cvrxlkitt63kg5uimeiwvgvenhwprfchsrzktjsikxqh5rt5wbfr.py
# Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6, add_9, sqrt_6, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_9 => add_9
#   hidden_states_15 => div_6
#   hidden_states_16 => convert_element_type_12, convert_element_type_13, mul_6, sigmoid_6
#   mean_6 => mean_6
#   output_tensor_2 => add_8
#   pow_7 => pow_7
#   sqrt_6 => sqrt_6
#   x_14 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_6, %arg13_1, %arg14_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %convolution_6), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_8, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [1], True), kwargs = {})
#   %add_9 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-08), kwargs = {})
#   %sqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_9,), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_8, %sqrt_6), kwargs = {})
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.float32), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_12,), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %sigmoid_6), kwargs = {})
#   %convert_element_type_13 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_6, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 105431040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 823680
    x0 = (xindex % 823680)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 128.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 873600*x1), tmp16, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/i2/ci2ygwsjrkmbjswkc2drmpk3cqkgpkclaiu4nxz4jsvfpoxkmkcx.py
# Topologically Sorted Source Nodes: [x_14, output_tensor_2, x_18, output_tensor_3], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_2 => add_8
#   output_tensor_3 => add_11
#   x_14 => convolution_6
#   x_18 => convolution_8
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_6, %arg13_1, %arg14_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_8 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_5, %convolution_6), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_8, %arg17_1, %arg18_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_11 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_8, %convolution_8), kwargs = {})
triton_poi_fused_add_convolution_14 = async_compile.triton('triton_poi_fused_add_convolution_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 105431040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 823680
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/vn/cvnmkc4jcbw7y2drdv565xu2gduhyypdsghk2i5tqb2lghmisc5j.py
# Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_19 => cat_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_9, %add_11], 2), kwargs = {})
triton_poi_fused_cat_15 = async_compile.triton('triton_poi_fused_cat_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 134217728}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_15(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 111820800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 24960) % 35)
    x0 = (xindex % 24960)
    x2 = xindex // 873600
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 823680*x2), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 35, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 24960*((-2) + x1) + 823680*x2), tmp6, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/uv/cuv2enlafgdtslqukq6epp6pt6gl2rcavkww2u7dparqb72sjonc.py
# Topologically Sorted Source Nodes: [x_19, x_20, x_26, pow_9, mean_8], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_8 => mean_8
#   pow_9 => pow_9
#   x_19 => cat_9
#   x_20 => convolution_9
#   x_26 => add_14, add_15, clone_6, convert_element_type_20, convert_element_type_21, mul_10, mul_11, rsqrt, sub, var_mean
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_9, %add_11], 2), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_9, %arg19_1, %arg20_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_20 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_6, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_20, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_20, %getitem_1), kwargs = {})
#   %add_14 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_14,), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %arg25_1), kwargs = {})
#   %add_15 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %arg26_1), kwargs = {})
#   %convert_element_type_21 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_15, torch.bfloat16), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_9, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [1], True), kwargs = {})
triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16 = async_compile.triton('triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 3, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 106080
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 106080*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
        tmp11 = tmp2 * tmp2
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (x0), tmp14, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(in_ptr0 + (x0 + 106080*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp5
        tmp21 = 128.0
        tmp22 = (tmp6 / tmp21)
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 128*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/gf/cgfe4ta6pg6qx2k5wmld5atnsjhivuaj37mrapmeu53rm56ynyll.py
# Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_tensor => convolution_12
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg27_1, %arg28_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_17 = async_compile.triton('triton_poi_fused_convolution_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 131072}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_17(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 106080
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 106080*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/4p/c4pg6ktxv3rko5herxmnf7rv2k2y5egcbcer64whgeftuhihg46k.py
# Topologically Sorted Source Nodes: [first_frame_pad_10], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_10 => repeat_10
# Graph fragment:
#   %repeat_10 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_53, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_18 = async_compile.triton('triton_poi_fused_repeat_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6240)
    x2 = xindex // 12480
    x3 = (xindex % 12480)
    tmp0 = tl.load(in_ptr0 + (x0 + 106080*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 128.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 118560*x2), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/mf/cmfsdnrrph7r35blt7dxeyfiuc5gxmonl4lt6mtavxz33ckzrjzp.py
# Topologically Sorted Source Nodes: [x_19, x_20, pow_9, mean_8, add_12, sqrt_8, hidden_states_20, hidden_states_21], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_12 => add_12
#   hidden_states_20 => div_8
#   hidden_states_21 => convert_element_type_16, convert_element_type_17, mul_8, sigmoid_8
#   mean_8 => mean_8
#   pow_9 => pow_9
#   sqrt_8 => sqrt_8
#   x_19 => cat_9
#   x_20 => convolution_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_9, %add_11], 2), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_9, %arg19_1, %arg20_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_9, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [1], True), kwargs = {})
#   %add_12 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_8, 1e-08), kwargs = {})
#   %sqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_12,), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_9, %sqrt_8), kwargs = {})
#   %convert_element_type_16 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.float32), kwargs = {})
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_16,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_16, %sigmoid_8), kwargs = {})
#   %convert_element_type_17 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_19 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 13578240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 106080
    x0 = (xindex % 106080)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 128.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 118560*x1), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/xn/cxnoqv66dxgrucb7babt3ejwsblbeuuujagf7ldw3buabmn4eiqy.py
# Topologically Sorted Source Nodes: [x_22, pow_10, mean_9], Original ATen: [aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_9 => mean_9
#   pow_10 => pow_10
#   x_22 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_10, %arg21_1, %arg22_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_10, 2), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [1], True), kwargs = {})
triton_red_fused_convolution_mean_pow_20 = async_compile.triton('triton_red_fused_convolution_mean_pow_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_pow_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_pow_20(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 106080
    r0_numel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + 106080*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/7m/c7msmvw3c3o7ezxyy5ad75gcmaxyiu5l63sl6lzkprr7qyv3zwic.py
# Topologically Sorted Source Nodes: [first_frame_pad_11], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_11 => repeat_11
# Graph fragment:
#   %repeat_11 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_58, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_21 = async_compile.triton('triton_poi_fused_repeat_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3194880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6240)
    x2 = xindex // 12480
    x3 = (xindex % 12480)
    tmp0 = tl.load(in_ptr0 + (x0 + 106080*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 256.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 118560*x2), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/xm/cxmqdo2q4ofemtzc233qoj4b7ppg5m2dmhgko74vuqlrehiqxoqs.py
# Topologically Sorted Source Nodes: [x_22, pow_10, mean_9, add_13, sqrt_9, hidden_states_22, hidden_states_23], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_13 => add_13
#   hidden_states_22 => div_9
#   hidden_states_23 => convert_element_type_18, convert_element_type_19, mul_9, sigmoid_9
#   mean_9 => mean_9
#   pow_10 => pow_10
#   sqrt_9 => sqrt_9
#   x_22 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_10, %arg21_1, %arg22_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_10, 2), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [1], True), kwargs = {})
#   %add_13 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_9, 1e-08), kwargs = {})
#   %sqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_13,), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_10, %sqrt_9), kwargs = {})
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_9, torch.float32), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_18,), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %sigmoid_9), kwargs = {})
#   %convert_element_type_19 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_9, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_22 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_22(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27156480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 106080
    x0 = (xindex % 106080)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 256.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 118560*x1), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/sp/cspahzfa6ktxs46w7gcro2nbatbjscvzc6vnkz3zgcvqqj663iy3.py
# Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   input_tensor => convolution_12
#   mean_10 => mean_10
#   output_tensor_4 => add_16
#   pow_11 => pow_11
#   x_24 => convolution_11
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg27_1, %arg28_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg23_1, %arg24_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_16, 2), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_11, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_23 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 106080
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 106080*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 106080*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/u3/cu3ofr7jr42oqsep4t2s56tpe4jwz5ic4bjsltronmeyuez435ra.py
# Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10, add_15, sqrt_10, hidden_states_25, hidden_states_26], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_15 => add_17
#   hidden_states_25 => div_10
#   hidden_states_26 => convert_element_type_22
#   input_tensor => convolution_12
#   mean_10 => mean_10
#   output_tensor_4 => add_16
#   pow_11 => pow_11
#   sqrt_10 => sqrt_10
#   x_24 => convolution_11
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg27_1, %arg28_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg23_1, %arg24_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_16, 2), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_11, [1], True), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_10, 1e-08), kwargs = {})
#   %sqrt_10 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_17,), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_16, %sqrt_10), kwargs = {})
#   %convert_element_type_22 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_10, torch.float32), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_24 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27156480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // 106080
    x0 = (xindex % 106080)
    tmp0 = tl.load(in_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = 256.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1e-08
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp15, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/me/cmea5smyqtizwdvbspph6oblkpkxm2wcmf6nmeo246moiiq62ahq.py
# Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_28 => cat_12
# Graph fragment:
#   %cat_12 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_12, %convert_element_type_23], 2), kwargs = {})
triton_poi_fused_cat_25 = async_compile.triton('triton_poi_fused_cat_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_25(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30351360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6240) % 19)
    x0 = (xindex % 6240)
    x2 = xindex // 118560
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 106080*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 19, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 6240*((-2) + x1) + 106080*x2), tmp11, other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/sh/csheepixi6wb2nvg4dsqodqfw55e4zjawlo2vxqk5tsezxxrlqtq.py
# Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, x_31, output_tensor_5], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_tensor => convolution_12
#   output_tensor_4 => add_16
#   output_tensor_5 => add_19
#   x_24 => convolution_11
#   x_31 => convolution_14
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg27_1, %arg28_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg23_1, %arg24_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_16 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_13, %arg31_1, %arg32_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_19 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_16, %convolution_14), kwargs = {})
triton_poi_fused_add_convolution_26 = async_compile.triton('triton_poi_fused_add_convolution_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27156480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = xindex // 106080
    x0 = (xindex % 6240)
    x4 = xindex // 6240
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (x0 + 6272*x4), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/tu/ctu2554xkymnbyf5vs7uih4v3ltfoppgavdb4litn73eycnq3oav.py
# Topologically Sorted Source Nodes: [pow_13, mean_12], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_12 => mean_12
#   pow_13 => pow_13
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_19, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_13, [1], True), kwargs = {})
triton_red_fused_mean_pow_27 = async_compile.triton('triton_red_fused_mean_pow_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_27(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 106080
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 6240)
    x1 = xindex // 6240
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 6272*x1 + 106624*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/do/cdocf4zfxyin4wwzxmifi3gluodicjum5suzzflkdhj5fwtewgqr.py
# Topologically Sorted Source Nodes: [first_frame_pad_14], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_14 => repeat_14
# Graph fragment:
#   %repeat_14 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_73, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_28 = async_compile.triton('triton_poi_fused_repeat_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_28(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 6240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 106624*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 256.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 1597440*y1), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/pr/cprttvofoiv75kh2jtztvvlizqn6xozscon6wbnqrm2hr7ve6yal.py
# Topologically Sorted Source Nodes: [pow_13, mean_12, add_18, sqrt_12, hidden_states_30, hidden_states_31], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_18 => add_20
#   hidden_states_30 => div_12
#   hidden_states_31 => convert_element_type_26, convert_element_type_27, mul_14, sigmoid_12
#   mean_12 => mean_12
#   pow_13 => pow_13
#   sqrt_12 => sqrt_12
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_19, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_13, [1], True), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_12, 1e-08), kwargs = {})
#   %sqrt_12 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_20,), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_19, %sqrt_12), kwargs = {})
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_12, torch.float32), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_26,), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, %sigmoid_12), kwargs = {})
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_14, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_29 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 131072}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_29(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 106080
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = (xindex % 6240)
    x2 = xindex // 6240
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 6272*x2 + 106624*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp2 = 256.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x3), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/eb/ceb7eyzegmdjgttgmzqwy6fesji3brtwfnaba6jedigariz5iwzp.py
# Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_33 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_14, %arg33_1, %arg34_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_30 = async_compile.triton('triton_poi_fused_convolution_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 65536, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_30(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 27
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 27*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 6912*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/dj/cdjvlyyookbeyjydl45kn2q4ignce237dtjc3535qp4lprcpwltf.py
# Topologically Sorted Source Nodes: [x_33, pow_14, mean_13, add_19, sqrt_13, hidden_states_32, hidden_states_33], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_19 => add_21
#   hidden_states_32 => div_13
#   hidden_states_33 => convert_element_type_28, convert_element_type_29, mul_15, sigmoid_13
#   mean_13 => mean_13
#   pow_14 => pow_14
#   sqrt_13 => sqrt_13
#   x_33 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_14, %arg33_1, %arg34_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_14 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_15, 2), kwargs = {})
#   %mean_13 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_14, [1], True), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_13, 1e-08), kwargs = {})
#   %sqrt_13 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_21,), kwargs = {})
#   %div_13 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_15, %sqrt_13), kwargs = {})
#   %convert_element_type_28 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_13, torch.float32), kwargs = {})
#   %sigmoid_13 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_28,), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_28, %sigmoid_13), kwargs = {})
#   %convert_element_type_29 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_15, torch.bfloat16), kwargs = {})
triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_31 = async_compile.triton('triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_31(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 106080
    r0_numel = 256
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = 256.0
        tmp12 = (tmp6 / tmp11)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = 1e-08
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.sqrt(tmp15)
        tmp17 = (tmp10 / tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.sigmoid(tmp18)
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr1 + (x0 + 118560*r0_1), tmp21, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/zs/czspezz6dl3cpnkbiyierg4uoejptxpzkyrsiohm62bw7e47tzrl.py
# Topologically Sorted Source Nodes: [first_frame_pad_15], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_15 => repeat_15
# Graph fragment:
#   %repeat_15 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_78, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_32 = async_compile.triton('triton_poi_fused_repeat_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12480
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = (xindex % 6240)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 256.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 118560*y0), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/wh/cwhspabj6oslzithm2uqqqrngkabq6nw2wvny5sm5k5cq2prhtul.py
# Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_14 => mean_14
#   output_tensor_6 => add_22
#   pow_15 => pow_15
#   x_35 => convolution_16
# Graph fragment:
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %arg35_1, %arg36_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %convolution_16), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_22, 2), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_15, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_33 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 131072, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_33', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_33(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 106080
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 6240)
    x1 = xindex // 6240
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 6272*x1 + 106624*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x3 + 106080*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/hk/chkbqejnmkhmwoi3z4hvzmkirw5g73g2cpuesfms6lff6dirhc56.py
# Topologically Sorted Source Nodes: [first_frame_pad_16], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_16 => repeat_16
# Graph fragment:
#   %repeat_16 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_83, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_34 = async_compile.triton('triton_poi_fused_repeat_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 6240
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 256)
    y1 = yindex // 256
    tmp0 = tl.load(in_ptr0 + (x2 + 106624*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 106080*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 256.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x2 + 1597440*y1), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/xx/cxxczvp3eebfpzs7pxahl2gu6hzbjjczgpw5cwoghfinsolscntk.py
# Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14, add_21, sqrt_14, hidden_states_35, hidden_states_36], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_21 => add_23
#   hidden_states_35 => div_14
#   hidden_states_36 => convert_element_type_30, convert_element_type_31, mul_16, sigmoid_14
#   mean_14 => mean_14
#   output_tensor_6 => add_22
#   pow_15 => pow_15
#   sqrt_14 => sqrt_14
#   x_35 => convolution_16
# Graph fragment:
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %arg35_1, %arg36_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %convolution_16), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_22, 2), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_15, [1], True), kwargs = {})
#   %add_23 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_14, 1e-08), kwargs = {})
#   %sqrt_14 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_23,), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_22, %sqrt_14), kwargs = {})
#   %convert_element_type_30 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_14, torch.float32), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_30,), kwargs = {})
#   %mul_16 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, %sigmoid_14), kwargs = {})
#   %convert_element_type_31 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_16, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_35 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 131072}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_35(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 106080
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = (xindex % 6240)
    x2 = xindex // 6240
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 6272*x2 + 106624*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3 + 106080*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x3), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 256.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x3), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/3u/c3uklsyjmtucm5iqtuligsf4x4ql5prlqfzb4v6ankdd25vf6jjm.py
# Topologically Sorted Source Nodes: [x_35, output_tensor_6, x_39, output_tensor_7], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_6 => add_22
#   output_tensor_7 => add_25
#   x_35 => convolution_16
#   x_39 => convolution_18
# Graph fragment:
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %arg35_1, %arg36_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_22 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_19, %convolution_16), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_17, %arg39_1, %arg40_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_25 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_22, %convolution_18), kwargs = {})
triton_poi_fused_add_convolution_36 = async_compile.triton('triton_poi_fused_add_convolution_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_36', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 27156480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6240)
    x3 = xindex // 6240
    x4 = xindex
    x2 = xindex // 106080
    tmp0 = tl.load(in_ptr0 + (x0 + 6272*x3), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/6g/c6gmolvlbk77hr4u5h77udj7h27ryunmrl7r6p2yxoemwgbkgeje.py
# Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_40 => cat_18
# Graph fragment:
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_18, %add_25], 2), kwargs = {})
triton_poi_fused_cat_37 = async_compile.triton('triton_poi_fused_cat_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_37(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30351360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6240) % 19)
    x0 = (xindex % 6240)
    x2 = xindex // 118560
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 106080*x2), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 19, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 6240*((-2) + x1) + 106080*x2), tmp6, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/kk/ckkitlp4m7pjku7lcsqfzr7fzjjit2fascdh2jjna7oz6q3rc3ht.py
# Topologically Sorted Source Nodes: [x_40, x_41, x_47, pow_17, mean_16], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_16 => mean_16
#   pow_17 => pow_17
#   x_40 => cat_18
#   x_41 => convolution_19
#   x_47 => add_28, add_29, clone_11, convert_element_type_38, convert_element_type_39, mul_20, mul_21, rsqrt_1, sub_1, var_mean_1
# Graph fragment:
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_18, %add_25], 2), kwargs = {})
#   %convolution_19 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_18, %arg41_1, %arg42_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_38 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_11, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [4]), kwargs = {correction: 0, keepdim: True})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_38, %getitem_3), kwargs = {})
#   %add_28 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt_1 : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_28,), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt_1), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %arg47_1), kwargs = {})
#   %add_29 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %arg48_1), kwargs = {})
#   %convert_element_type_39 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_29, torch.bfloat16), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_19, 2), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [1], True), kwargs = {})
triton_red_fused_cat_convolution_mean_native_layer_norm_pow_38 = async_compile.triton('triton_red_fused_cat_convolution_mean_native_layer_norm_pow_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr2': '*fp32', 'out_ptr3': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_native_layer_norm_pow_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 3, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_native_layer_norm_pow_38(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 14040
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp5_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp5_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x2 = (xindex % 1560)
    x3 = xindex // 1560
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 14040*r0_1), r0_mask & xmask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp5_mean_next, tmp5_m2_next, tmp5_weight_next = triton_helpers.welford_reduce(
            tmp4, tmp5_mean, tmp5_m2, tmp5_weight, roffset == 0
        )
        tmp5_mean = tl.where(r0_mask & xmask, tmp5_mean_next, tmp5_mean)
        tmp5_m2 = tl.where(r0_mask & xmask, tmp5_m2_next, tmp5_m2)
        tmp5_weight = tl.where(r0_mask & xmask, tmp5_weight_next, tmp5_weight)
        tmp11 = tmp2 * tmp2
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(r0_mask & xmask, tmp15, _tmp14)
    tmp8, tmp9, tmp10 = triton_helpers.welford(tmp5_mean, tmp5_m2, tmp5_weight, 1)
    tmp5 = tmp8[:, None]
    tmp6 = tmp9[:, None]
    tmp7 = tmp10[:, None]
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr2 + (x2 + 1568*x3), tmp14, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp16 = tl.load(in_ptr0 + (x0 + 14040*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp17 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp27 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp30 = tl.load(in_ptr3 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp19 - tmp5
        tmp21 = 256.0
        tmp22 = (tmp6 / tmp21)
        tmp23 = 1e-06
        tmp24 = tmp22 + tmp23
        tmp25 = libdevice.rsqrt(tmp24)
        tmp26 = tmp20 * tmp25
        tmp28 = tmp27.to(tl.float32)
        tmp29 = tmp26 * tmp28
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp29 + tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr3 + (r0_1 + 256*x0), tmp33, r0_mask & xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/u5/cu5bdubvf5wp2pavz6aby3ujsshubr43boub5kgulmi6mwd2kqpt.py
# Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   input_tensor_1 => convolution_22
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg49_1, %arg50_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_39 = async_compile.triton('triton_poi_fused_convolution_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 16384}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_39(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 14040
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 14040*y0), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/m5/cm5lpg4qokv6es3fddckh2pgnw3z3ezqh2g5y4gmxgospwcdpfzm.py
# Topologically Sorted Source Nodes: [first_frame_pad_19], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_19 => repeat_19
# Graph fragment:
#   %repeat_19 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_98, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_40 = async_compile.triton('triton_poi_fused_repeat_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 798720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1560)
    x2 = xindex // 3120
    x3 = (xindex % 3120)
    tmp0 = tl.load(in_ptr0 + (x0 + 14040*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 256.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 17160*x2), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/yc/cycdhlduvvajdf4fcyenu6w6cseqloxsusskz73cbmetcl6huag7.py
# Topologically Sorted Source Nodes: [x_40, x_41, pow_17, mean_16, add_24, sqrt_16, hidden_states_40, hidden_states_41], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_24 => add_26
#   hidden_states_40 => div_16
#   hidden_states_41 => convert_element_type_34, convert_element_type_35, mul_18, sigmoid_16
#   mean_16 => mean_16
#   pow_17 => pow_17
#   sqrt_16 => sqrt_16
#   x_40 => cat_18
#   x_41 => convolution_19
# Graph fragment:
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_18, %add_25], 2), kwargs = {})
#   %convolution_19 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_18, %arg41_1, %arg42_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_19, 2), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [1], True), kwargs = {})
#   %add_26 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_16, 1e-08), kwargs = {})
#   %sqrt_16 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_26,), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_19, %sqrt_16), kwargs = {})
#   %convert_element_type_34 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_16, torch.float32), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_34,), kwargs = {})
#   %mul_18 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_34, %sigmoid_16), kwargs = {})
#   %convert_element_type_35 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_18, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_41 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3594240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x2 = xindex // 14040
    x0 = (xindex % 1560)
    x1 = ((xindex // 1560) % 9)
    x4 = (xindex % 14040)
    tmp0 = tl.load(in_ptr0 + (x3), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 1568*x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 256.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x4 + 17160*x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/cq/ccq2rrmwfklzennk3fvtqs5u3zz2ljdwaiurvmfephlhyypt6np3.py
# Topologically Sorted Source Nodes: [x_43, pow_18, mean_17], Original ATen: [aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_17 => mean_17
#   pow_18 => pow_18
#   x_43 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_19, %arg43_1, %arg44_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_18 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_20, 2), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_18, [1], True), kwargs = {})
triton_red_fused_convolution_mean_pow_42 = async_compile.triton('triton_red_fused_convolution_mean_pow_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_pow_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_pow_42(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 14040
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x3 + 14040*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/d5/cd5wa6xd2u2vo5yjdidmvgyh7wckyhcnumwgjfekwy5y2aknyhxj.py
# Topologically Sorted Source Nodes: [first_frame_pad_20], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_20 => repeat_20
# Graph fragment:
#   %repeat_20 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_103, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_43 = async_compile.triton('triton_poi_fused_repeat_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_43(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1560)
    x2 = xindex // 3120
    x3 = (xindex % 3120)
    tmp0 = tl.load(in_ptr0 + (x0 + 14040*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 512.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 17160*x2), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/pi/cpimil2ix6ctr3j6p4vzrfx7h6yvevswh4h2l4mcovecnevkkcmc.py
# Topologically Sorted Source Nodes: [x_43, pow_18, mean_17, add_25, sqrt_17, hidden_states_42, hidden_states_43], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_25 => add_27
#   hidden_states_42 => div_17
#   hidden_states_43 => convert_element_type_36, convert_element_type_37, mul_19, sigmoid_17
#   mean_17 => mean_17
#   pow_18 => pow_18
#   sqrt_17 => sqrt_17
#   x_43 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_19, %arg43_1, %arg44_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_18 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_20, 2), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_18, [1], True), kwargs = {})
#   %add_27 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_17, 1e-08), kwargs = {})
#   %sqrt_17 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_27,), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_20, %sqrt_17), kwargs = {})
#   %convert_element_type_36 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_17, torch.float32), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_36,), kwargs = {})
#   %mul_19 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %sigmoid_17), kwargs = {})
#   %convert_element_type_37 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_19, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_44 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_44', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_44(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7188480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = xindex // 14040
    x0 = (xindex % 1560)
    x1 = ((xindex // 1560) % 9)
    x4 = (xindex % 14040)
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0 + 1568*x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 512.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x4 + 17160*x2), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/li/cliikbb5tgb6d7tz4773xv6g3c6x565zp3jwovdwseclbdfjpvkh.py
# Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   input_tensor_1 => convolution_22
#   mean_18 => mean_18
#   output_tensor_8 => add_30
#   pow_19 => pow_19
#   x_45 => convolution_21
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg49_1, %arg50_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_20, %arg45_1, %arg46_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_30 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_30, 2), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_19, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_45 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 14040
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x3 + 14040*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x3 + 14040*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/pj/cpj3kwfsa5yabjgvvdlwa4ozio5f5yv4i22gc5wecpu7txg2x5oy.py
# Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18, add_27, sqrt_18, hidden_states_45, hidden_states_46], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_31
#   hidden_states_45 => div_18
#   hidden_states_46 => convert_element_type_40
#   input_tensor_1 => convolution_22
#   mean_18 => mean_18
#   output_tensor_8 => add_30
#   pow_19 => pow_19
#   sqrt_18 => sqrt_18
#   x_45 => convolution_21
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg49_1, %arg50_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_20, %arg45_1, %arg46_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_30 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_30, 2), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_19, [1], True), kwargs = {})
#   %add_31 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_18, 1e-08), kwargs = {})
#   %sqrt_18 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_31,), kwargs = {})
#   %div_18 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_30, %sqrt_18), kwargs = {})
#   %convert_element_type_40 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_18, torch.float32), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_46 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_46(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7188480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = xindex // 14040
    x0 = (xindex % 1560)
    x1 = ((xindex // 1560) % 9)
    x4 = xindex // 1560
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x0 + 1568*x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = 512.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1e-08
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 1568*x4), tmp15, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/5v/c5vqudq6yefabrpygmyltobd25yt5ihvlxoidzlmi6ymgepqqny3.py
# Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_49 => cat_21
# Graph fragment:
#   %cat_21 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_21, %convert_element_type_41], 2), kwargs = {})
triton_poi_fused_cat_47 = async_compile.triton('triton_poi_fused_cat_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_47(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8785920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1560) % 11)
    x0 = (xindex % 1560)
    x2 = xindex // 17160
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 14112*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 11, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 1568*((-2) + x1) + 14112*x2), tmp11, other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/qu/cqukppf2anbrys6knlijamuqtehzb3kbt2ihzkxdwcu54cqlvq3p.py
# Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, x_52, output_tensor_9], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_tensor_1 => convolution_22
#   output_tensor_8 => add_30
#   output_tensor_9 => add_33
#   x_45 => convolution_21
#   x_52 => convolution_24
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg49_1, %arg50_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_20, %arg45_1, %arg46_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_30 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_22, %arg53_1, %arg54_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_33 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_30, %convolution_24), kwargs = {})
triton_poi_fused_add_convolution_48 = async_compile.triton('triton_poi_fused_add_convolution_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7188480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x3 = xindex
    x2 = xindex // 14040
    x0 = (xindex % 1560)
    x4 = xindex // 1560
    tmp0 = tl.load(in_ptr0 + (x3), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x3), None).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x3), None).to(tl.float32)
    tmp8 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (x0 + 1600*x4), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/i5/ci5xvwn2k7hjictklxvigjgtbcsuyyn2ovecc43otbbr62dhwk5v.py
# Topologically Sorted Source Nodes: [pow_21, mean_20], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_20 => mean_20
#   pow_21 => pow_21
# Graph fragment:
#   %pow_21 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_33, 2), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_21, [1], True), kwargs = {})
triton_red_fused_mean_pow_49 = async_compile.triton('triton_red_fused_mean_pow_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_49(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 14040
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1600*x1 + 14400*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ss/css4zuhpfqnp7mbi3elakafzq5zushuoqvsqqtwee65xjo2egr2u.py
# Topologically Sorted Source Nodes: [first_frame_pad_23], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_23 => repeat_23
# Graph fragment:
#   %repeat_23 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_118, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_50 = async_compile.triton('triton_poi_fused_repeat_50', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_50(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1560
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 14400*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = 512.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 798720*y1), tmp12, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/c6/cc6rafygh4jc3by4qyirvo2osjkxurovqbvkuwbmrjimseipsflf.py
# Topologically Sorted Source Nodes: [pow_21, mean_20, add_30, sqrt_20, hidden_states_50, hidden_states_51], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_30 => add_34
#   hidden_states_50 => div_20
#   hidden_states_51 => convert_element_type_44, convert_element_type_45, mul_24, sigmoid_20
#   mean_20 => mean_20
#   pow_21 => pow_21
#   sqrt_20 => sqrt_20
# Graph fragment:
#   %pow_21 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_33, 2), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_21, [1], True), kwargs = {})
#   %add_34 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_20, 1e-08), kwargs = {})
#   %sqrt_20 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_34,), kwargs = {})
#   %div_20 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_33, %sqrt_20), kwargs = {})
#   %convert_element_type_44 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_20, torch.float32), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_44,), kwargs = {})
#   %mul_24 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_44, %sigmoid_20), kwargs = {})
#   %convert_element_type_45 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_24, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_51 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_51(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 14040
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = (xindex % 1560)
    x2 = xindex // 1560
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1600*x2 + 14400*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + 1568*x2), xmask, eviction_policy='evict_last')
    tmp2 = 512.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x3), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/du/cduinv57373ngdovarkul3btvutvmz33yeoxmbkc2duwyn7xhdro.py
# Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_54 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_23, %arg55_1, %arg56_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_52 = async_compile.triton('triton_poi_fused_convolution_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 262144, 'x': 32}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2DWithYZOverflow', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_52', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_52(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 27
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 27*y3), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 13824*y1), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/n4/cn4noagqlgnurlj6nw2iz3lzp2ghltsdxgola3q4emxrqkbq3xkc.py
# Topologically Sorted Source Nodes: [x_54, pow_22, mean_21, add_31, sqrt_21, hidden_states_52, hidden_states_53], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_31 => add_35
#   hidden_states_52 => div_21
#   hidden_states_53 => convert_element_type_46, convert_element_type_47, mul_25, sigmoid_21
#   mean_21 => mean_21
#   pow_22 => pow_22
#   sqrt_21 => sqrt_21
#   x_54 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_23, %arg55_1, %arg56_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_22 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_25, 2), kwargs = {})
#   %mean_21 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_22, [1], True), kwargs = {})
#   %add_35 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_21, 1e-08), kwargs = {})
#   %sqrt_21 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_35,), kwargs = {})
#   %div_21 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_25, %sqrt_21), kwargs = {})
#   %convert_element_type_46 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_21, torch.float32), kwargs = {})
#   %sigmoid_21 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_46,), kwargs = {})
#   %mul_25 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_46, %sigmoid_21), kwargs = {})
#   %convert_element_type_47 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_25, torch.bfloat16), kwargs = {})
triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_53 = async_compile.triton('triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_53(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 14040
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 512*x3), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp6, xmask)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp8 = tl.load(in_ptr0 + (r0_2 + 512*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = 512.0
        tmp12 = (tmp6 / tmp11)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = 1e-08
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.sqrt(tmp15)
        tmp17 = (tmp10 / tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tl.sigmoid(tmp18)
        tmp20 = tmp18 * tmp19
        tmp21 = tmp20.to(tl.float32)
        tl.store(out_ptr1 + (x3 + 17160*r0_2), tmp21, xmask & r0_mask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ga/cga4xixaknaai4lwytv2za3jgywv257cxgq245dncueftw3k7kxu.py
# Topologically Sorted Source Nodes: [first_frame_pad_24], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_24 => repeat_24
# Graph fragment:
#   %repeat_24 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_123, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_54 = async_compile.triton('triton_poi_fused_repeat_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 4096}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_54(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3120
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = (xindex % 1560)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 512.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 17160*y0), tmp14, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/c4/cc46jqbwdgyiuuxnbdkp4a5zlxuincyt2exh4zuxu2tj3wc4kz7b.py
# Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_22 => mean_22
#   output_tensor_10 => add_36
#   pow_23 => pow_23
#   x_56 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_24, %arg57_1, %arg58_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_36 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %convolution_26), kwargs = {})
#   %pow_23 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_36, 2), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_23, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_55 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16384, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_55(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 14040
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1560)
    x1 = xindex // 1560
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1600*x1 + 14400*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x3 + 14040*r0_2), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1568*x1), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/pd/cpdtaz66y6lgml6gt2s2nehm6ospbyxbhqrvwopak6zwdn4oxzqd.py
# Topologically Sorted Source Nodes: [first_frame_pad_25], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_25 => repeat_25
# Graph fragment:
#   %repeat_25 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_128, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_56 = async_compile.triton('triton_poi_fused_repeat_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_56(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 1560
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 512)
    y1 = yindex // 512
    tmp0 = tl.load(in_ptr0 + (x2 + 14400*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 14040*y0), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 512.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x2 + 798720*y1), tmp16, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/io/ciodxg5t36acwwkxqvefq276tvdqtyzo3w6rleubhnq7kfqxsoat.py
# Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22, add_33, sqrt_22, hidden_states_55, hidden_states_56], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_33 => add_37
#   hidden_states_55 => div_22
#   hidden_states_56 => convert_element_type_48, convert_element_type_49, mul_26, sigmoid_22
#   mean_22 => mean_22
#   output_tensor_10 => add_36
#   pow_23 => pow_23
#   sqrt_22 => sqrt_22
#   x_56 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_24, %arg57_1, %arg58_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_36 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %convolution_26), kwargs = {})
#   %pow_23 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_36, 2), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_23, [1], True), kwargs = {})
#   %add_37 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_22, 1e-08), kwargs = {})
#   %sqrt_22 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_37,), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_36, %sqrt_22), kwargs = {})
#   %convert_element_type_48 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_22, torch.float32), kwargs = {})
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_48,), kwargs = {})
#   %mul_26 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %sigmoid_22), kwargs = {})
#   %convert_element_type_49 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_26, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_57 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 16384}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 14040
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = (xindex % 1560)
    x2 = xindex // 1560
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1600*x2 + 14400*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x3 + 14040*y0), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x1 + 1568*x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 512.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 512*x3), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/dq/cdqsu24k5ehd7f2uaymionojnsstqhw3scvbjthhxcarrrgyzrps.py
# Topologically Sorted Source Nodes: [x_56, output_tensor_10, x_60, output_tensor_11], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_10 => add_36
#   output_tensor_11 => add_39
#   x_56 => convolution_26
#   x_60 => convolution_28
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_24, %arg57_1, %arg58_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_36 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_33, %convolution_26), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_26, %arg61_1, %arg62_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_39 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_36, %convolution_28), kwargs = {})
triton_poi_fused_add_convolution_58 = async_compile.triton('triton_poi_fused_add_convolution_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_58', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_58(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7188480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1560)
    x3 = xindex // 1560
    x4 = xindex
    x2 = xindex // 14040
    tmp0 = tl.load(in_ptr0 + (x0 + 1600*x3), None).to(tl.float32)
    tmp1 = tl.load(in_out_ptr0 + (x4), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x4), None).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x4), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/iw/ciwd74ufwhwht4qbyxpoqnwt2h3qpatrhmpvjftanezv6vt2upkg.py
# Topologically Sorted Source Nodes: [x_61], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_61 => cat_27
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_39], 2), kwargs = {})
triton_poi_fused_cat_59 = async_compile.triton('triton_poi_fused_cat_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_59(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8785920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1560) % 11)
    x0 = (xindex % 1560)
    x2 = xindex // 17160
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 14040*x2), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 11, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 1560*((-2) + x1) + 14040*x2), tmp6, other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/g7/cg7umvtyridnq63zprwgyarmarrqymnv7ec4e74q4kecqptazipf.py
# Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_24 => mean_24
#   pow_25 => pow_25
#   x_61 => cat_27
#   x_62 => convolution_29
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_39], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg63_1, %arg64_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_29, 2), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_25, [1], True), kwargs = {})
triton_red_fused_cat_convolution_mean_pow_60 = async_compile.triton('triton_red_fused_cat_convolution_mean_pow_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_pow_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_pow_60(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 7800
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1950)
    x1 = xindex // 1950
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1950*r0_2 + 249600*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1952*x1), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/zz/czzqrcynp4wiix2rkirkchqefhiyxjpsx3zvm6tm4wgvqij3rlay.py
# Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_24 => mean_24
#   pow_25 => pow_25
#   x_61 => cat_27
#   x_62 => convolution_29
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_39], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg63_1, %arg64_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_29, 2), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_25, [1], True), kwargs = {})
triton_per_fused_cat_convolution_mean_pow_61 = async_compile.triton('triton_per_fused_cat_convolution_mean_pow_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 2048, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_convolution_mean_pow_61', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_convolution_mean_pow_61(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1950
    r0_numel = 4
    R0_BLOCK: tl.constexpr = 4
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1952*r0_1), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ab/cabnhj746apru3hvxvhnex2bfe5a52clj5y3gsqmipgfz7sr64go.py
# Topologically Sorted Source Nodes: [first_frame_pad_28], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_28 => repeat_28
# Graph fragment:
#   %repeat_28 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_143, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_62 = async_compile.triton('triton_poi_fused_repeat_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_62(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 399360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 390)
    x2 = xindex // 780
    x3 = (xindex % 780)
    tmp0 = tl.load(in_ptr0 + (x0 + 1950*x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 512.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 2730*x2), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/34/c34bljkpaooxcmgxjkslgvt6ct2qw6burev7fdgwcnvnzaupvqna.py
# Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24, add_36, sqrt_24, hidden_states_60, hidden_states_61], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_36 => add_40
#   hidden_states_60 => div_24
#   hidden_states_61 => convert_element_type_52, convert_element_type_53, mul_28, sigmoid_24
#   mean_24 => mean_24
#   pow_25 => pow_25
#   sqrt_24 => sqrt_24
#   x_61 => cat_27
#   x_62 => convolution_29
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_39], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg63_1, %arg64_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_29, 2), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_25, [1], True), kwargs = {})
#   %add_40 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_24, 1e-08), kwargs = {})
#   %sqrt_24 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_40,), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_29, %sqrt_24), kwargs = {})
#   %convert_element_type_52 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_24, torch.float32), kwargs = {})
#   %sigmoid_24 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_52,), kwargs = {})
#   %mul_28 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %sigmoid_24), kwargs = {})
#   %convert_element_type_53 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_28, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1950
    x0 = (xindex % 1950)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = 512.0
    tmp5 = (tmp3 / tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = 1e-08
    tmp8 = tmp6 + tmp7
    tmp9 = libdevice.sqrt(tmp8)
    tmp10 = (tmp2 / tmp9)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 2730*x1), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/fb/cfbuqzhjaybnu5oaektgigi6izbdapm2liiuyrz2qbw6pslpk6mc.py
# Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_26 => mean_26
#   output_tensor_12 => add_42
#   pow_27 => pow_27
#   x_61 => cat_27
#   x_62 => convolution_29
#   x_66 => convolution_31
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_39], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg63_1, %arg64_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg67_1, %arg68_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_42 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, %convolution_31), kwargs = {})
#   %pow_27 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_42, 2), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [1], True), kwargs = {})
triton_red_fused_add_cat_convolution_mean_pow_64 = async_compile.triton('triton_red_fused_add_cat_convolution_mean_pow_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_convolution_mean_pow_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_convolution_mean_pow_64(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 7800
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1950)
    x1 = xindex // 1950
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1950*r0_2 + 249600*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 1950*r0_2 + 249600*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 + tmp5
        tmp7 = tmp6 * tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(r0_mask & xmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1952*x1), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/4u/c4un5usv7e2gbwsagfpdttezsdjt7mjc77lljnveu2toyfrwfpde.py
# Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26, add_39, sqrt_26, hidden_states_65, hidden_states_66], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_39 => add_43
#   hidden_states_65 => div_26
#   hidden_states_66 => convert_element_type_56
#   mean_26 => mean_26
#   output_tensor_12 => add_42
#   pow_27 => pow_27
#   sqrt_26 => sqrt_26
#   x_61 => cat_27
#   x_62 => convolution_29
#   x_66 => convolution_31
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_39], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg63_1, %arg64_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg67_1, %arg68_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_42 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, %convolution_31), kwargs = {})
#   %pow_27 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_42, 2), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [1], True), kwargs = {})
#   %add_43 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_26, 1e-08), kwargs = {})
#   %sqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_43,), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_42, %sqrt_26), kwargs = {})
#   %convert_element_type_56 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_26, torch.float32), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_65 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_65(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1950
    x0 = (xindex % 1950)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp8 = 512.0
    tmp9 = (tmp7 / tmp8)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = 1e-08
    tmp12 = tmp10 + tmp11
    tmp13 = libdevice.sqrt(tmp12)
    tmp14 = (tmp6 / tmp13)
    tmp15 = tmp14.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 1952*x1), tmp15, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ia/cia3uztrpymmvgooancffz5mgi7t4vcwvqtzu3i2inud2qqiw66n.py
# Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   x_67 => cat_30
# Graph fragment:
#   %cat_30 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_30, %convert_element_type_57], 2), kwargs = {})
triton_poi_fused_cat_66 = async_compile.triton('triton_poi_fused_cat_66', '''
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
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_66(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1397760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 390) % 7)
    x0 = (xindex % 390)
    x2 = xindex // 2730
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1952*x2), xmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 7, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 390*((-2) + x1) + 1952*x2), xmask & tmp11, other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/yh/cyhrzwf6geq2m2qal52n3kgsiziw3azeawqrsqvqdikjsdlptog6.py
# Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, x_70, output_tensor_13], Original ATen: [aten.cat, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_12 => add_42
#   output_tensor_13 => add_45
#   x_61 => cat_27
#   x_62 => convolution_29
#   x_66 => convolution_31
#   x_70 => convolution_33
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_39], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg63_1, %arg64_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg67_1, %arg68_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_42 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, %convolution_31), kwargs = {})
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_31, %arg71_1, %arg72_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_45 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_42, %convolution_33), kwargs = {})
triton_poi_fused_add_cat_convolution_67 = async_compile.triton('triton_poi_fused_add_cat_convolution_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // 1950
    x0 = (xindex % 1950)
    tmp0 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), xmask).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x2), xmask).to(tl.float32)
    tmp8 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(out_ptr0 + (x0 + 1984*x1), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/mb/cmbs5kjvn422ompkwhgpiswo5ud4e7juixdxojrx7p4nz6fl5d6t.py
# Topologically Sorted Source Nodes: [pow_29, mean_28], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_28 => mean_28
#   pow_29 => pow_29
# Graph fragment:
#   %pow_29 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_45, 2), kwargs = {})
#   %mean_28 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_29, [1], True), kwargs = {})
triton_red_fused_mean_pow_68 = async_compile.triton('triton_red_fused_mean_pow_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_68', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_68(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 7800
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1950)
    x1 = xindex // 1950
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1984*r0_2 + 253952*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1952*x1), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/oh/cohnzuzfyxtrf562eiw3db7jsvwob6mtmcbporj3b3mw77uc3adv.py
# Topologically Sorted Source Nodes: [first_frame_pad_32], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_32 => repeat_32
# Graph fragment:
#   %repeat_32 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_163, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_69 = async_compile.triton('triton_poi_fused_repeat_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_69(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 399360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 390)
    x2 = xindex // 780
    x3 = (xindex % 780)
    tmp0 = tl.load(in_ptr0 + (x0 + 1984*x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 512.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 2730*x2), tmp12, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/7u/c7uvpgywgv3c3nxbro62qhr5vmaisrkvh2oqnaphmslup2wvlj2p.py
# Topologically Sorted Source Nodes: [pow_29, mean_28, add_42, sqrt_28, hidden_states_70, hidden_states_71], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_42 => add_46
#   hidden_states_70 => div_28
#   hidden_states_71 => convert_element_type_60, convert_element_type_61, mul_32, sigmoid_28
#   mean_28 => mean_28
#   pow_29 => pow_29
#   sqrt_28 => sqrt_28
# Graph fragment:
#   %pow_29 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_45, 2), kwargs = {})
#   %mean_28 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_29, [1], True), kwargs = {})
#   %add_46 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_28, 1e-08), kwargs = {})
#   %sqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_46,), kwargs = {})
#   %div_28 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_45, %sqrt_28), kwargs = {})
#   %convert_element_type_60 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_28, torch.float32), kwargs = {})
#   %sigmoid_28 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_60,), kwargs = {})
#   %mul_32 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_60, %sigmoid_28), kwargs = {})
#   %convert_element_type_61 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_32, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_70 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_70(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1950)
    x1 = xindex // 1950
    tmp0 = tl.load(in_ptr0 + (x0 + 1984*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = 512.0
    tmp3 = (tmp1 / tmp2)
    tmp4 = tmp3.to(tl.float32)
    tmp5 = 1e-08
    tmp6 = tmp4 + tmp5
    tmp7 = libdevice.sqrt(tmp6)
    tmp8 = (tmp0 / tmp7)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 2730*x1), tmp12, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/g2/cg274rrov4ha7kvqgx3ehqz6ps67od64rmxhmhfinqx3za7rvkbt.py
# Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_30 => mean_30
#   output_tensor_14 => add_48
#   pow_31 => pow_31
#   x_74 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %arg75_1, %arg76_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %convolution_35), kwargs = {})
#   %pow_31 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_48, 2), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_31, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_71 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_71', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_71', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_71(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 7800
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 1950)
    x1 = xindex // 1950
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1984*r0_2 + 253952*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 1950*r0_2 + 249600*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tmp4 * tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0 + 1952*x1), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/qk/cqkcpdngiso7nsgcpdtpmgpb2b3nly26t6kmdvu3jo7hmxcvljjx.py
# Topologically Sorted Source Nodes: [first_frame_pad_34], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_34 => repeat_34
# Graph fragment:
#   %repeat_34 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_173, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_72 = async_compile.triton('triton_poi_fused_repeat_72', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_72', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_72(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 399360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 390)
    x2 = xindex // 780
    x3 = (xindex % 780)
    tmp0 = tl.load(in_ptr0 + (x0 + 1984*x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 1950*x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 512.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 2730*x2), tmp16, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/gm/cgmkrkqrlr476cisgw2vf567q5gzmtvsgsufd6v6ebitub4dmjtn.py
# Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30, add_45, sqrt_30, hidden_states_75, hidden_states_76], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_45 => add_49
#   hidden_states_75 => div_30
#   hidden_states_76 => convert_element_type_64, convert_element_type_65, mul_34, sigmoid_30
#   mean_30 => mean_30
#   output_tensor_14 => add_48
#   pow_31 => pow_31
#   sqrt_30 => sqrt_30
#   x_74 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %arg75_1, %arg76_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %convolution_35), kwargs = {})
#   %pow_31 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_48, 2), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_31, [1], True), kwargs = {})
#   %add_49 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_30, 1e-08), kwargs = {})
#   %sqrt_30 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_49,), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_48, %sqrt_30), kwargs = {})
#   %convert_element_type_64 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_30, torch.float32), kwargs = {})
#   %sigmoid_30 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_64,), kwargs = {})
#   %mul_34 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, %sigmoid_30), kwargs = {})
#   %convert_element_type_65 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_34, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_73 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_73', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_73', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_73(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1950)
    x1 = xindex // 1950
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 1984*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = 512.0
    tmp7 = (tmp5 / tmp6)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = 1e-08
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.sqrt(tmp10)
    tmp12 = (tmp4 / tmp11)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = tmp15.to(tl.float32)
    tl.store(out_ptr0 + (x0 + 2730*x1), tmp16, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/h2/ch2u7uspdsdrdyrbqc2blfv4i2aunpyrahjwq3ou37gtg3grtbfu.py
# Topologically Sorted Source Nodes: [x_74, output_tensor_14, x_78, output_tensor_15], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_14 => add_48
#   output_tensor_15 => add_51
#   x_74 => convolution_35
#   x_78 => convolution_37
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %arg75_1, %arg76_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_48 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_45, %convolution_35), kwargs = {})
#   %convolution_37 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_35, %arg79_1, %arg80_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_51 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_48, %convolution_37), kwargs = {})
triton_poi_fused_add_convolution_74 = async_compile.triton('triton_poi_fused_add_convolution_74', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_74', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_74(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 998400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1950)
    x1 = xindex // 1950
    x2 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 1984*x1), xmask).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), xmask).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x0 + 1984*x1), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/5b/c5b5cwou2fdkbyu3zmaoawejlm7nxnjq62fyshu2gach6ktawbh3.py
# Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   sample_2 => cat_43
# Graph fragment:
#   %cat_43 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_44, %repeat_43], 1), kwargs = {})
triton_poi_fused_cat_75 = async_compile.triton('triton_poi_fused_cat_75', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_75', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_75(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 499200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // 1950
    x0 = (xindex % 1950)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 129, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1950*(x1)), xmask & tmp4, other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x1), xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (249600 + x0), xmask & tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (128)).to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp16 = tl.where(tmp10, tmp15, 0.0)
    tmp17 = tmp13 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp10, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp9, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 3, 33, 832, 480), (39536640, 13178880, 399360, 480, 1))
    assert_size_stride(arg1_1, (128, 48, 3, 3, 3), (1296, 27, 9, 3, 1))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (256, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg24_1, (256, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg28_1, (256, ), (1, ))
    assert_size_stride(arg29_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg30_1, (256, ), (1, ))
    assert_size_stride(arg31_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg34_1, (256, ), (1, ))
    assert_size_stride(arg35_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg38_1, (256, ), (1, ))
    assert_size_stride(arg39_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg42_1, (256, ), (1, ))
    assert_size_stride(arg43_1, (512, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg46_1, (512, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg50_1, (512, ), (1, ))
    assert_size_stride(arg51_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg54_1, (512, ), (1, ))
    assert_size_stride(arg55_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg56_1, (512, ), (1, ))
    assert_size_stride(arg57_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg58_1, (512, ), (1, ))
    assert_size_stride(arg59_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg60_1, (512, ), (1, ))
    assert_size_stride(arg61_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg64_1, (512, ), (1, ))
    assert_size_stride(arg65_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg66_1, (512, ), (1, ))
    assert_size_stride(arg67_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg68_1, (512, ), (1, ))
    assert_size_stride(arg69_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg72_1, (512, ), (1, ))
    assert_size_stride(arg73_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg74_1, (512, ), (1, ))
    assert_size_stride(arg75_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg76_1, (512, ), (1, ))
    assert_size_stride(arg77_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg78_1, (512, ), (1, ))
    assert_size_stride(arg79_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg80_1, (512, ), (1, ))
    assert_size_stride(arg81_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg82_1, (512, ), (1, ))
    assert_size_stride(arg83_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg84_1, (512, ), (1, ))
    assert_size_stride(arg85_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg86_1, (512, ), (1, ))
    assert_size_stride(arg87_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg88_1, (512, ), (1, ))
    assert_size_stride(arg89_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg90_1, (512, ), (1, ))
    assert_size_stride(arg91_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (129, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg94_1, (129, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((1, 48, 35, 208, 120), (41932800, 873600, 24960, 120, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_0.run(arg0_1, buf0, 41932800, stream=stream0)
        del arg0_1
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.cat, aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg1_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg1_1
        del buf0
        buf2 = empty_strided_cuda((1, 1, 33, 208, 120), (823680, 823680, 24960, 120, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf1, arg2_1, buf2, 823680, 128, stream=stream0)
        buf5 = empty_strided_cuda((1, 128, 35, 208, 120), (111820800, 873600, 24960, 120, 1), torch.bfloat16)
        buf3 = reinterpret_tensor(buf5, (1, 128, 2, 208, 120), (111820800, 873600, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_1], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf1, arg2_1, buf2, buf3, 6389760, stream=stream0)
        buf4 = reinterpret_tensor(buf5, (1, 128, 33, 208, 120), (111820800, 873600, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean, add, sqrt, hidden_states, hidden_states_1], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf1, arg2_1, buf2, buf4, 105431040, stream=stream0)
        del buf3
        del buf4
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg3_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg3_1
        buf7 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_4, pow_2, mean_1], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf6, arg4_1, buf7, 823680, 128, stream=stream0)
        buf10 = buf5; del buf5  # reuse
        buf8 = reinterpret_tensor(buf10, (1, 128, 2, 208, 120), (111820800, 873600, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_2], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf6, arg4_1, buf7, buf8, 6389760, stream=stream0)
        buf9 = reinterpret_tensor(buf10, (1, 128, 33, 208, 120), (111820800, 873600, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_4, pow_2, mean_1, add_1, sqrt_1, hidden_states_2, hidden_states_3], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf6, arg4_1, buf7, buf9, 105431040, stream=stream0)
        del arg4_1
        del buf6
        del buf8
        del buf9
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg5_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg5_1
        buf12 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_convolution_mean_pow_4.run(buf1, arg2_1, buf11, arg6_1, buf12, 823680, 128, stream=stream0)
        buf13 = empty_strided_cuda((1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2, add_3, sqrt_2, hidden_states_5, hidden_states_6], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5.run(buf1, arg2_1, buf11, arg6_1, buf12, buf13, 105431040, stream=stream0)
        buf14 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_7], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_6.run(buf13, buf14, 111820800, stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.cat, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg7_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg7_1
        buf16 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_8, pow_4, mean_3], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf15, arg8_1, buf16, 823680, 128, stream=stream0)
        buf19 = buf14; del buf14  # reuse
        buf17 = reinterpret_tensor(buf19, (1, 128, 2, 208, 120), (111820800, 873600, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_4], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf15, arg8_1, buf16, buf17, 6389760, stream=stream0)
        buf18 = reinterpret_tensor(buf19, (1, 128, 33, 208, 120), (111820800, 873600, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_7, x_8, pow_4, mean_3, add_4, sqrt_3, hidden_states_7, hidden_states_8], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf15, arg8_1, buf16, buf18, 105431040, stream=stream0)
        del arg8_1
        del buf15
        del buf17
        del buf18
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg9_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg9_1
        buf21 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, x_10, output_tensor_1], Original ATen: [aten.cat, aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_7.run(buf21, arg2_1, buf11, arg6_1, buf20, arg10_1, 105431040, stream=stream0)
        del arg10_1
        del arg2_1
        del arg6_1
        del buf11
        del buf20
        buf22 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [pow_5, mean_4], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_8.run(buf21, buf22, 823680, 128, stream=stream0)
        buf25 = buf19; del buf19  # reuse
        buf23 = reinterpret_tensor(buf25, (1, 128, 2, 208, 120), (111820800, 873600, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_5], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_9.run(buf21, buf22, buf23, 6389760, stream=stream0)
        buf24 = reinterpret_tensor(buf25, (1, 128, 33, 208, 120), (111820800, 873600, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [pow_5, mean_4, add_6, sqrt_4, hidden_states_10, hidden_states_11], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_10.run(buf21, buf22, buf24, 105431040, stream=stream0)
        del buf23
        del buf24
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg11_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg11_1
        buf27 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_12, pow_6, mean_5], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf26, arg12_1, buf27, 823680, 128, stream=stream0)
        buf30 = buf25; del buf25  # reuse
        buf28 = reinterpret_tensor(buf30, (1, 128, 2, 208, 120), (111820800, 873600, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_6], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf26, arg12_1, buf27, buf28, 6389760, stream=stream0)
        buf29 = reinterpret_tensor(buf30, (1, 128, 33, 208, 120), (111820800, 873600, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_12, pow_6, mean_5, add_7, sqrt_5, hidden_states_12, hidden_states_13], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf26, arg12_1, buf27, buf29, 105431040, stream=stream0)
        del arg12_1
        del buf26
        del buf28
        del buf29
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg13_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg13_1
        buf32 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_11.run(buf21, buf31, arg14_1, buf32, 823680, 128, stream=stream0)
        buf35 = buf30; del buf30  # reuse
        buf33 = reinterpret_tensor(buf35, (1, 128, 2, 208, 120), (111820800, 873600, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_7], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_12.run(buf21, buf31, arg14_1, buf32, buf33, 6389760, stream=stream0)
        buf34 = reinterpret_tensor(buf35, (1, 128, 33, 208, 120), (111820800, 873600, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6, add_9, sqrt_6, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13.run(buf21, buf31, arg14_1, buf32, buf34, 105431040, stream=stream0)
        del buf33
        del buf34
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg15_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg15_1
        buf37 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_16, pow_8, mean_7], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf36, arg16_1, buf37, 823680, 128, stream=stream0)
        buf40 = buf35; del buf35  # reuse
        buf38 = reinterpret_tensor(buf40, (1, 128, 2, 208, 120), (111820800, 873600, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_8], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf36, arg16_1, buf37, buf38, 6389760, stream=stream0)
        buf39 = reinterpret_tensor(buf40, (1, 128, 33, 208, 120), (111820800, 873600, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_16, pow_8, mean_7, add_10, sqrt_7, hidden_states_17, hidden_states_18], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf36, arg16_1, buf37, buf39, 105431040, stream=stream0)
        del arg16_1
        del buf36
        del buf37
        del buf38
        del buf39
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg17_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (1, 128, 33, 208, 120), (105431040, 823680, 24960, 120, 1))
        del arg17_1
        buf42 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_14, output_tensor_2, x_18, output_tensor_3], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_14.run(buf42, buf31, arg14_1, buf41, arg18_1, 105431040, stream=stream0)
        del arg14_1
        del arg18_1
        del buf31
        del buf41
        buf43 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_19], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_15.run(buf42, buf43, 111820800, stream=stream0)
        del buf42
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten.cat, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg19_1, stride=(2, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (1, 128, 17, 104, 60), (13578240, 106080, 6240, 60, 1))
        del arg19_1
        del buf43
        buf51 = empty_strided_cuda((1, 1, 17, 104, 60), (106080, 106080, 6240, 60, 1), torch.float32)
        buf48 = empty_strided_cuda((1, 17, 104, 60, 128), (13578240, 798720, 7680, 128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_19, x_20, x_26, pow_9, mean_8], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16.run(buf44, arg20_1, arg25_1, arg26_1, buf51, buf48, 106080, 128, stream=stream0)
        del arg25_1
        del arg26_1
        buf49 = empty_strided_cuda((1, 128, 17, 104, 60), (13578240, 106080, 6240, 60, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_17.run(buf48, buf49, 128, 106080, stream=stream0)
        del buf48
        # Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg27_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1))
        del arg27_1
        del buf49
        buf54 = empty_strided_cuda((1, 128, 19, 104, 60), (15175680, 118560, 6240, 60, 1), torch.bfloat16)
        buf52 = reinterpret_tensor(buf54, (1, 128, 2, 104, 60), (15175680, 118560, 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_10], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_18.run(buf44, arg20_1, buf51, buf52, 1597440, stream=stream0)
        buf53 = reinterpret_tensor(buf54, (1, 128, 17, 104, 60), (15175680, 118560, 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_19, x_20, pow_9, mean_8, add_12, sqrt_8, hidden_states_20, hidden_states_21], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_19.run(buf44, arg20_1, buf51, buf53, 13578240, stream=stream0)
        del arg20_1
        del buf44
        del buf52
        del buf53
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, arg21_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1))
        del arg21_1
        del buf54
        buf56 = buf51; del buf51  # reuse
        # Topologically Sorted Source Nodes: [x_22, pow_10, mean_9], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_20.run(buf55, arg22_1, buf56, 106080, 256, stream=stream0)
        buf59 = empty_strided_cuda((1, 256, 19, 104, 60), (30351360, 118560, 6240, 60, 1), torch.bfloat16)
        buf57 = reinterpret_tensor(buf59, (1, 256, 2, 104, 60), (30351360, 118560, 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_11], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_21.run(buf55, arg22_1, buf56, buf57, 3194880, stream=stream0)
        buf58 = reinterpret_tensor(buf59, (1, 256, 17, 104, 60), (30351360, 118560, 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_22, pow_10, mean_9, add_13, sqrt_9, hidden_states_22, hidden_states_23], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_22.run(buf55, arg22_1, buf56, buf58, 27156480, stream=stream0)
        del arg22_1
        del buf55
        del buf57
        del buf58
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg23_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1))
        del arg23_1
        buf61 = buf56; del buf56  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_23.run(buf50, arg28_1, buf60, arg24_1, buf61, 106080, 256, stream=stream0)
        buf62 = empty_strided_cuda((1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10, add_15, sqrt_10, hidden_states_25, hidden_states_26], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_24.run(buf50, arg28_1, buf60, arg24_1, buf61, buf62, 27156480, stream=stream0)
        buf63 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [x_28], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_25.run(buf62, buf63, 30351360, stream=stream0)
        del buf62
        # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.cat, aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg29_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1))
        del arg29_1
        buf65 = buf61; del buf61  # reuse
        # Topologically Sorted Source Nodes: [x_28, x_29, pow_12, mean_11], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_20.run(buf64, arg30_1, buf65, 106080, 256, stream=stream0)
        buf68 = buf63; del buf63  # reuse
        buf66 = reinterpret_tensor(buf68, (1, 256, 2, 104, 60), (30351360, 118560, 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_13], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_21.run(buf64, arg30_1, buf65, buf66, 3194880, stream=stream0)
        buf67 = reinterpret_tensor(buf68, (1, 256, 17, 104, 60), (30351360, 118560, 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_28, x_29, pow_12, mean_11, add_16, sqrt_11, hidden_states_27, hidden_states_28], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_22.run(buf64, arg30_1, buf65, buf67, 27156480, stream=stream0)
        del arg30_1
        del buf64
        del buf66
        del buf67
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg31_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1))
        del arg31_1
        buf70 = empty_strided_cuda((1, 256, 17, 104, 60), (27295744, 106624, 6272, 60, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, x_31, output_tensor_5], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_26.run(buf50, arg28_1, buf60, arg24_1, buf69, arg32_1, buf70, 27156480, stream=stream0)
        del arg24_1
        del arg28_1
        del arg32_1
        del buf50
        del buf60
        del buf69
        buf71 = buf65; del buf65  # reuse
        # Topologically Sorted Source Nodes: [pow_13, mean_12], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_27.run(buf70, buf71, 106080, 256, stream=stream0)
        buf74 = reinterpret_tensor(buf68, (1, 256, 19, 104, 60), (30351360, 1, 1597440, 15360, 256), 0); del buf68  # reuse
        buf72 = reinterpret_tensor(buf74, (1, 256, 2, 104, 60), (30351360, 1, 1597440, 15360, 256), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_14], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_28.run(buf70, buf71, buf72, 512, 6240, stream=stream0)
        buf73 = reinterpret_tensor(buf74, (1, 256, 17, 104, 60), (30351360, 1, 1597440, 15360, 256), 3194880)  # alias
        # Topologically Sorted Source Nodes: [pow_13, mean_12, add_18, sqrt_12, hidden_states_30, hidden_states_31], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_29.run(buf70, buf71, buf73, 256, 106080, stream=stream0)
        buf75 = empty_strided_cuda((256, 256, 3, 3, 3), (6912, 1, 2304, 768, 256), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_30.run(arg33_1, buf75, 65536, 27, stream=stream0)
        del arg33_1
        del buf72
        del buf73
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf74, buf75, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (1, 256, 17, 104, 60), (27156480, 1, 1597440, 15360, 256))
        buf77 = buf71; del buf71  # reuse
        buf80 = reinterpret_tensor(buf74, (1, 256, 19, 104, 60), (30351360, 118560, 6240, 60, 1), 0); del buf74  # reuse
        buf79 = reinterpret_tensor(buf80, (1, 256, 17, 104, 60), (30351360, 118560, 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_33, pow_14, mean_13, add_19, sqrt_13, hidden_states_32, hidden_states_33], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_31.run(buf76, arg34_1, buf77, buf79, 106080, 256, stream=stream0)
        buf78 = reinterpret_tensor(buf80, (1, 256, 2, 104, 60), (30351360, 118560, 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_15], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_32.run(buf76, arg34_1, buf77, buf78, 256, 12480, stream=stream0)
        del arg34_1
        del buf76
        del buf78
        del buf79
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg35_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1))
        del arg35_1
        buf82 = buf77; del buf77  # reuse
        # Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_33.run(buf70, buf81, arg36_1, buf82, 106080, 256, stream=stream0)
        buf85 = reinterpret_tensor(buf80, (1, 256, 19, 104, 60), (30351360, 1, 1597440, 15360, 256), 0); del buf80  # reuse
        buf83 = reinterpret_tensor(buf85, (1, 256, 2, 104, 60), (30351360, 1, 1597440, 15360, 256), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_16], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_34.run(buf70, buf81, arg36_1, buf82, buf83, 512, 6240, stream=stream0)
        buf84 = reinterpret_tensor(buf85, (1, 256, 17, 104, 60), (30351360, 1, 1597440, 15360, 256), 3194880)  # alias
        # Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14, add_21, sqrt_14, hidden_states_35, hidden_states_36], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_35.run(buf70, buf81, arg36_1, buf82, buf84, 256, 106080, stream=stream0)
        buf86 = buf75; del buf75  # reuse
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_30.run(arg37_1, buf86, 65536, 27, stream=stream0)
        del arg37_1
        del buf83
        del buf84
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf85, buf86, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (1, 256, 17, 104, 60), (27156480, 1, 1597440, 15360, 256))
        del buf86
        buf88 = buf82; del buf82  # reuse
        buf91 = reinterpret_tensor(buf85, (1, 256, 19, 104, 60), (30351360, 118560, 6240, 60, 1), 0); del buf85  # reuse
        buf90 = reinterpret_tensor(buf91, (1, 256, 17, 104, 60), (30351360, 118560, 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_37, pow_16, mean_15, add_22, sqrt_15, hidden_states_37, hidden_states_38], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_31.run(buf87, arg38_1, buf88, buf90, 106080, 256, stream=stream0)
        buf89 = reinterpret_tensor(buf91, (1, 256, 2, 104, 60), (30351360, 118560, 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_17], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_32.run(buf87, arg38_1, buf88, buf89, 256, 12480, stream=stream0)
        del arg38_1
        del buf87
        del buf88
        del buf89
        del buf90
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg39_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (1, 256, 17, 104, 60), (27156480, 106080, 6240, 60, 1))
        del arg39_1
        buf93 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_35, output_tensor_6, x_39, output_tensor_7], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_36.run(buf93, buf70, arg36_1, buf92, arg40_1, 27156480, stream=stream0)
        del arg36_1
        del arg40_1
        del buf70
        del buf92
        buf94 = buf91; del buf91  # reuse
        # Topologically Sorted Source Nodes: [x_40], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_37.run(buf93, buf94, 30351360, stream=stream0)
        del buf93
        # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.cat, aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg41_1, stride=(2, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (1, 256, 9, 52, 30), (3594240, 14040, 1560, 30, 1))
        del arg41_1
        del buf94
        buf102 = empty_strided_cuda((1, 1, 9, 52, 30), (14112, 14112, 1568, 30, 1), torch.float32)
        buf99 = empty_strided_cuda((1, 9, 52, 30, 256), (3594240, 399360, 7680, 256, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_40, x_41, x_47, pow_17, mean_16], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_native_layer_norm_pow_38.run(buf95, arg42_1, arg47_1, arg48_1, buf102, buf99, 14040, 256, stream=stream0)
        del arg47_1
        del arg48_1
        buf100 = empty_strided_cuda((1, 256, 9, 52, 30), (3594240, 14040, 1560, 30, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_39.run(buf99, buf100, 256, 14040, stream=stream0)
        del buf99
        # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg49_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (1, 512, 9, 52, 30), (7188480, 14040, 1560, 30, 1))
        del arg49_1
        del buf100
        buf105 = empty_strided_cuda((1, 256, 11, 52, 30), (4392960, 17160, 1560, 30, 1), torch.bfloat16)
        buf103 = reinterpret_tensor(buf105, (1, 256, 2, 52, 30), (4392960, 17160, 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_19], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_40.run(buf95, arg42_1, buf102, buf103, 798720, stream=stream0)
        buf104 = reinterpret_tensor(buf105, (1, 256, 9, 52, 30), (4392960, 17160, 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_40, x_41, pow_17, mean_16, add_24, sqrt_16, hidden_states_40, hidden_states_41], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_41.run(buf95, arg42_1, buf102, buf104, 3594240, stream=stream0)
        del arg42_1
        del buf95
        del buf103
        del buf104
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg43_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (1, 512, 9, 52, 30), (7188480, 14040, 1560, 30, 1))
        del arg43_1
        del buf105
        buf107 = buf102; del buf102  # reuse
        # Topologically Sorted Source Nodes: [x_43, pow_18, mean_17], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_42.run(buf106, arg44_1, buf107, 14040, 512, stream=stream0)
        buf110 = empty_strided_cuda((1, 512, 11, 52, 30), (8785920, 17160, 1560, 30, 1), torch.bfloat16)
        buf108 = reinterpret_tensor(buf110, (1, 512, 2, 52, 30), (8785920, 17160, 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_20], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_43.run(buf106, arg44_1, buf107, buf108, 1597440, stream=stream0)
        buf109 = reinterpret_tensor(buf110, (1, 512, 9, 52, 30), (8785920, 17160, 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_43, pow_18, mean_17, add_25, sqrt_17, hidden_states_42, hidden_states_43], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_44.run(buf106, arg44_1, buf107, buf109, 7188480, stream=stream0)
        del arg44_1
        del buf106
        del buf108
        del buf109
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg45_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (1, 512, 9, 52, 30), (7188480, 14040, 1560, 30, 1))
        del arg45_1
        buf112 = buf107; del buf107  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_45.run(buf101, arg50_1, buf111, arg46_1, buf112, 14040, 512, stream=stream0)
        buf113 = empty_strided_cuda((1, 512, 9, 52, 30), (7225344, 14112, 1568, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18, add_27, sqrt_18, hidden_states_45, hidden_states_46], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_46.run(buf101, arg50_1, buf111, arg46_1, buf112, buf113, 7188480, stream=stream0)
        buf114 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_49], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_47.run(buf113, buf114, 8785920, stream=stream0)
        del buf113
        # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.cat, aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg51_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (1, 512, 9, 52, 30), (7188480, 14040, 1560, 30, 1))
        del arg51_1
        buf116 = buf112; del buf112  # reuse
        # Topologically Sorted Source Nodes: [x_49, x_50, pow_20, mean_19], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_42.run(buf115, arg52_1, buf116, 14040, 512, stream=stream0)
        buf119 = buf114; del buf114  # reuse
        buf117 = reinterpret_tensor(buf119, (1, 512, 2, 52, 30), (8785920, 17160, 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_22], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_43.run(buf115, arg52_1, buf116, buf117, 1597440, stream=stream0)
        buf118 = reinterpret_tensor(buf119, (1, 512, 9, 52, 30), (8785920, 17160, 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_49, x_50, pow_20, mean_19, add_28, sqrt_19, hidden_states_47, hidden_states_48], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_44.run(buf115, arg52_1, buf116, buf118, 7188480, stream=stream0)
        del arg52_1
        del buf115
        del buf117
        del buf118
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, arg53_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (1, 512, 9, 52, 30), (7188480, 14040, 1560, 30, 1))
        del arg53_1
        buf121 = empty_strided_cuda((1, 512, 9, 52, 30), (7372800, 14400, 1600, 30, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, x_52, output_tensor_9], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_48.run(buf101, arg50_1, buf111, arg46_1, buf120, arg54_1, buf121, 7188480, stream=stream0)
        del arg46_1
        del arg50_1
        del arg54_1
        del buf101
        del buf111
        del buf120
        buf122 = buf116; del buf116  # reuse
        # Topologically Sorted Source Nodes: [pow_21, mean_20], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_49.run(buf121, buf122, 14040, 512, stream=stream0)
        buf125 = reinterpret_tensor(buf119, (1, 512, 11, 52, 30), (8785920, 1, 798720, 15360, 512), 0); del buf119  # reuse
        buf123 = reinterpret_tensor(buf125, (1, 512, 2, 52, 30), (8785920, 1, 798720, 15360, 512), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_23], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_50.run(buf121, buf122, buf123, 1024, 1560, stream=stream0)
        buf124 = reinterpret_tensor(buf125, (1, 512, 9, 52, 30), (8785920, 1, 798720, 15360, 512), 1597440)  # alias
        # Topologically Sorted Source Nodes: [pow_21, mean_20, add_30, sqrt_20, hidden_states_50, hidden_states_51], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_51.run(buf121, buf122, buf124, 512, 14040, stream=stream0)
        buf126 = empty_strided_cuda((512, 512, 3, 3, 3), (13824, 1, 4608, 1536, 512), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(arg55_1, buf126, 262144, 27, stream=stream0)
        del arg55_1
        del buf123
        del buf124
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf125, buf126, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (1, 512, 9, 52, 30), (7188480, 1, 798720, 15360, 512))
        buf128 = buf122; del buf122  # reuse
        buf131 = reinterpret_tensor(buf125, (1, 512, 11, 52, 30), (8785920, 17160, 1560, 30, 1), 0); del buf125  # reuse
        buf130 = reinterpret_tensor(buf131, (1, 512, 9, 52, 30), (8785920, 17160, 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_54, pow_22, mean_21, add_31, sqrt_21, hidden_states_52, hidden_states_53], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_53.run(buf127, arg56_1, buf128, buf130, 14040, 512, stream=stream0)
        buf129 = reinterpret_tensor(buf131, (1, 512, 2, 52, 30), (8785920, 17160, 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_24], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_54.run(buf127, arg56_1, buf128, buf129, 512, 3120, stream=stream0)
        del arg56_1
        del buf127
        del buf129
        del buf130
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, arg57_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (1, 512, 9, 52, 30), (7188480, 14040, 1560, 30, 1))
        del arg57_1
        buf133 = buf128; del buf128  # reuse
        # Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_55.run(buf121, buf132, arg58_1, buf133, 14040, 512, stream=stream0)
        buf136 = reinterpret_tensor(buf131, (1, 512, 11, 52, 30), (8785920, 1, 798720, 15360, 512), 0); del buf131  # reuse
        buf134 = reinterpret_tensor(buf136, (1, 512, 2, 52, 30), (8785920, 1, 798720, 15360, 512), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_25], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf121, buf132, arg58_1, buf133, buf134, 1024, 1560, stream=stream0)
        buf135 = reinterpret_tensor(buf136, (1, 512, 9, 52, 30), (8785920, 1, 798720, 15360, 512), 1597440)  # alias
        # Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22, add_33, sqrt_22, hidden_states_55, hidden_states_56], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_57.run(buf121, buf132, arg58_1, buf133, buf135, 512, 14040, stream=stream0)
        buf137 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_52.run(arg59_1, buf137, 262144, 27, stream=stream0)
        del arg59_1
        del buf134
        del buf135
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf138 = extern_kernels.convolution(buf136, buf137, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf138, (1, 512, 9, 52, 30), (7188480, 1, 798720, 15360, 512))
        del buf137
        buf139 = buf133; del buf133  # reuse
        buf142 = reinterpret_tensor(buf136, (1, 512, 11, 52, 30), (8785920, 17160, 1560, 30, 1), 0); del buf136  # reuse
        buf141 = reinterpret_tensor(buf142, (1, 512, 9, 52, 30), (8785920, 17160, 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_58, pow_24, mean_23, add_34, sqrt_23, hidden_states_57, hidden_states_58], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_div_mean_pow_silu_sqrt_53.run(buf138, arg60_1, buf139, buf141, 14040, 512, stream=stream0)
        buf140 = reinterpret_tensor(buf142, (1, 512, 2, 52, 30), (8785920, 17160, 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_26], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_54.run(buf138, arg60_1, buf139, buf140, 512, 3120, stream=stream0)
        del arg60_1
        del buf138
        del buf139
        del buf140
        del buf141
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, arg61_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (1, 512, 9, 52, 30), (7188480, 14040, 1560, 30, 1))
        del arg61_1
        buf144 = buf132; del buf132  # reuse
        # Topologically Sorted Source Nodes: [x_56, output_tensor_10, x_60, output_tensor_11], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_58.run(buf144, buf121, arg58_1, buf143, arg62_1, 7188480, stream=stream0)
        del arg58_1
        del arg62_1
        del buf121
        del buf143
        buf145 = buf142; del buf142  # reuse
        # Topologically Sorted Source Nodes: [x_61], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_59.run(buf144, buf145, 8785920, stream=stream0)
        del buf144
        # Topologically Sorted Source Nodes: [x_61, x_62], Original ATen: [aten.cat, aten.convolution]
        buf146 = extern_kernels.convolution(buf145, arg63_1, stride=(2, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg63_1
        del buf145
        buf147 = empty_strided_cuda((1, 1, 5, 26, 15, 4), (7808, 7808, 390, 15, 1, 1952), torch.float32)
        # Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf146, arg64_1, buf147, 7800, 128, stream=stream0)
        buf148 = empty_strided_cuda((1, 1, 5, 26, 15), (1952, 1952, 390, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf147, buf148, 1950, 4, stream=stream0)
        buf151 = empty_strided_cuda((1, 512, 7, 26, 15), (1397760, 2730, 390, 15, 1), torch.bfloat16)
        buf149 = reinterpret_tensor(buf151, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_28], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf146, arg64_1, buf148, buf149, 399360, stream=stream0)
        buf150 = reinterpret_tensor(buf151, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24, add_36, sqrt_24, hidden_states_60, hidden_states_61], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf146, arg64_1, buf148, buf150, 998400, stream=stream0)
        del buf149
        del buf150
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, arg65_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg65_1
        buf153 = buf147; del buf147  # reuse
        # Topologically Sorted Source Nodes: [x_64, pow_26, mean_25], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf152, arg66_1, buf153, 7800, 128, stream=stream0)
        buf154 = buf148; del buf148  # reuse
        # Topologically Sorted Source Nodes: [x_64, pow_26, mean_25], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf153, buf154, 1950, 4, stream=stream0)
        buf157 = buf151; del buf151  # reuse
        buf155 = reinterpret_tensor(buf157, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_29], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf152, arg66_1, buf154, buf155, 399360, stream=stream0)
        buf156 = reinterpret_tensor(buf157, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_64, pow_26, mean_25, add_37, sqrt_25, hidden_states_62, hidden_states_63], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf152, arg66_1, buf154, buf156, 998400, stream=stream0)
        del arg66_1
        del buf152
        del buf155
        del buf156
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf157, arg67_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg67_1
        buf159 = buf153; del buf153  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_convolution_mean_pow_64.run(buf146, arg64_1, buf158, arg68_1, buf159, 7800, 128, stream=stream0)
        buf160 = buf154; del buf154  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf159, buf160, 1950, 4, stream=stream0)
        buf161 = empty_strided_cuda((1, 512, 5, 26, 15), (999424, 1952, 390, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26, add_39, sqrt_26, hidden_states_65, hidden_states_66], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_65.run(buf146, arg64_1, buf158, arg68_1, buf160, buf161, 998400, stream=stream0)
        buf162 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_67], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_66.run(buf161, buf162, 1397760, stream=stream0)
        del buf161
        # Topologically Sorted Source Nodes: [x_67, x_68], Original ATen: [aten.cat, aten.convolution]
        buf163 = extern_kernels.convolution(buf162, arg69_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg69_1
        buf164 = buf159; del buf159  # reuse
        # Topologically Sorted Source Nodes: [x_67, x_68, pow_28, mean_27], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf163, arg70_1, buf164, 7800, 128, stream=stream0)
        buf165 = buf160; del buf160  # reuse
        # Topologically Sorted Source Nodes: [x_67, x_68, pow_28, mean_27], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf164, buf165, 1950, 4, stream=stream0)
        buf168 = buf162; del buf162  # reuse
        buf166 = reinterpret_tensor(buf168, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_31], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf163, arg70_1, buf165, buf166, 399360, stream=stream0)
        buf167 = reinterpret_tensor(buf168, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_67, x_68, pow_28, mean_27, add_40, sqrt_27, hidden_states_67, hidden_states_68], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf163, arg70_1, buf165, buf167, 998400, stream=stream0)
        del arg70_1
        del buf163
        del buf166
        del buf167
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, arg71_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg71_1
        buf170 = empty_strided_cuda((1, 512, 5, 26, 15), (1015808, 1984, 390, 15, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, x_70, output_tensor_13], Original ATen: [aten.cat, aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_67.run(buf146, arg64_1, buf158, arg68_1, buf169, arg72_1, buf170, 998400, stream=stream0)
        del arg64_1
        del arg68_1
        del arg72_1
        del buf146
        del buf158
        del buf169
        buf171 = buf164; del buf164  # reuse
        # Topologically Sorted Source Nodes: [pow_29, mean_28], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_68.run(buf170, buf171, 7800, 128, stream=stream0)
        buf172 = buf165; del buf165  # reuse
        # Topologically Sorted Source Nodes: [pow_29, mean_28], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf171, buf172, 1950, 4, stream=stream0)
        buf175 = buf168; del buf168  # reuse
        buf173 = reinterpret_tensor(buf175, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_32], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_69.run(buf170, buf172, buf173, 399360, stream=stream0)
        buf174 = reinterpret_tensor(buf175, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [pow_29, mean_28, add_42, sqrt_28, hidden_states_70, hidden_states_71], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_70.run(buf170, buf172, buf174, 998400, stream=stream0)
        del buf173
        del buf174
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf175, arg73_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg73_1
        buf177 = buf171; del buf171  # reuse
        # Topologically Sorted Source Nodes: [x_72, pow_30, mean_29], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf176, arg74_1, buf177, 7800, 128, stream=stream0)
        buf178 = buf172; del buf172  # reuse
        # Topologically Sorted Source Nodes: [x_72, pow_30, mean_29], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf177, buf178, 1950, 4, stream=stream0)
        buf181 = buf175; del buf175  # reuse
        buf179 = reinterpret_tensor(buf181, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_33], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf176, arg74_1, buf178, buf179, 399360, stream=stream0)
        buf180 = reinterpret_tensor(buf181, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_72, pow_30, mean_29, add_43, sqrt_29, hidden_states_72, hidden_states_73], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf176, arg74_1, buf178, buf180, 998400, stream=stream0)
        del arg74_1
        del buf176
        del buf179
        del buf180
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, arg75_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg75_1
        buf183 = buf177; del buf177  # reuse
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_71.run(buf170, buf182, arg76_1, buf183, 7800, 128, stream=stream0)
        buf184 = buf178; del buf178  # reuse
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf183, buf184, 1950, 4, stream=stream0)
        buf187 = buf181; del buf181  # reuse
        buf185 = reinterpret_tensor(buf187, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_34], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_72.run(buf170, buf182, arg76_1, buf184, buf185, 399360, stream=stream0)
        buf186 = reinterpret_tensor(buf187, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30, add_45, sqrt_30, hidden_states_75, hidden_states_76], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_73.run(buf170, buf182, arg76_1, buf184, buf186, 998400, stream=stream0)
        del buf185
        del buf186
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, arg77_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg77_1
        buf189 = buf183; del buf183  # reuse
        # Topologically Sorted Source Nodes: [x_76, pow_32, mean_31], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf188, arg78_1, buf189, 7800, 128, stream=stream0)
        buf190 = buf184; del buf184  # reuse
        # Topologically Sorted Source Nodes: [x_76, pow_32, mean_31], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf189, buf190, 1950, 4, stream=stream0)
        buf193 = buf187; del buf187  # reuse
        buf191 = reinterpret_tensor(buf193, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_35], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf188, arg78_1, buf190, buf191, 399360, stream=stream0)
        buf192 = reinterpret_tensor(buf193, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_76, pow_32, mean_31, add_46, sqrt_31, hidden_states_77, hidden_states_78], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf188, arg78_1, buf190, buf192, 998400, stream=stream0)
        del arg78_1
        del buf188
        del buf191
        del buf192
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, arg79_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg79_1
        buf195 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, x_78, output_tensor_15], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_74.run(buf195, buf182, arg76_1, buf194, arg80_1, 998400, stream=stream0)
        del arg76_1
        del arg80_1
        del buf182
        del buf194
        buf196 = buf189; del buf189  # reuse
        # Topologically Sorted Source Nodes: [pow_33, mean_32], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_68.run(buf195, buf196, 7800, 128, stream=stream0)
        buf197 = buf190; del buf190  # reuse
        # Topologically Sorted Source Nodes: [pow_33, mean_32], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf196, buf197, 1950, 4, stream=stream0)
        buf200 = buf193; del buf193  # reuse
        buf198 = reinterpret_tensor(buf200, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_36], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_69.run(buf195, buf197, buf198, 399360, stream=stream0)
        buf199 = reinterpret_tensor(buf200, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [pow_33, mean_32, add_48, sqrt_32, hidden_states_80, hidden_states_81], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_70.run(buf195, buf197, buf199, 998400, stream=stream0)
        del buf198
        del buf199
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, arg81_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg81_1
        buf202 = buf196; del buf196  # reuse
        # Topologically Sorted Source Nodes: [x_80, pow_34, mean_33], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf201, arg82_1, buf202, 7800, 128, stream=stream0)
        buf203 = buf197; del buf197  # reuse
        # Topologically Sorted Source Nodes: [x_80, pow_34, mean_33], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf202, buf203, 1950, 4, stream=stream0)
        buf206 = buf200; del buf200  # reuse
        buf204 = reinterpret_tensor(buf206, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_37], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf201, arg82_1, buf203, buf204, 399360, stream=stream0)
        buf205 = reinterpret_tensor(buf206, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_80, pow_34, mean_33, add_49, sqrt_33, hidden_states_82, hidden_states_83], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf201, arg82_1, buf203, buf205, 998400, stream=stream0)
        del arg82_1
        del buf201
        del buf204
        del buf205
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf207 = extern_kernels.convolution(buf206, arg83_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg83_1
        buf208 = buf202; del buf202  # reuse
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, pow_35, mean_34], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_71.run(buf195, buf207, arg84_1, buf208, 7800, 128, stream=stream0)
        buf209 = buf203; del buf203  # reuse
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, pow_35, mean_34], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf208, buf209, 1950, 4, stream=stream0)
        buf212 = buf206; del buf206  # reuse
        buf210 = reinterpret_tensor(buf212, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_38], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_72.run(buf195, buf207, arg84_1, buf209, buf210, 399360, stream=stream0)
        buf211 = reinterpret_tensor(buf212, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, pow_35, mean_34, add_51, sqrt_34, hidden_states_85, hidden_states_86], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_73.run(buf195, buf207, arg84_1, buf209, buf211, 998400, stream=stream0)
        del buf210
        del buf211
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, arg85_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg85_1
        buf214 = buf208; del buf208  # reuse
        # Topologically Sorted Source Nodes: [x_84, pow_36, mean_35], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf213, arg86_1, buf214, 7800, 128, stream=stream0)
        buf215 = buf209; del buf209  # reuse
        # Topologically Sorted Source Nodes: [x_84, pow_36, mean_35], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf214, buf215, 1950, 4, stream=stream0)
        buf218 = buf212; del buf212  # reuse
        buf216 = reinterpret_tensor(buf218, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_39], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf213, arg86_1, buf215, buf216, 399360, stream=stream0)
        buf217 = reinterpret_tensor(buf218, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_84, pow_36, mean_35, add_52, sqrt_35, hidden_states_87, hidden_states_88], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf213, arg86_1, buf215, buf217, 998400, stream=stream0)
        del arg86_1
        del buf213
        del buf216
        del buf217
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg87_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg87_1
        buf220 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, x_86, output_tensor_17], Original ATen: [aten.convolution, aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_74.run(buf220, buf207, arg84_1, buf219, arg88_1, 998400, stream=stream0)
        del arg84_1
        del arg88_1
        del buf207
        del buf219
        buf221 = buf214; del buf214  # reuse
        # Topologically Sorted Source Nodes: [pow_37, mean_36], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_68.run(buf220, buf221, 7800, 128, stream=stream0)
        buf222 = buf215; del buf215  # reuse
        # Topologically Sorted Source Nodes: [pow_37, mean_36], Original ATen: [aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf221, buf222, 1950, 4, stream=stream0)
        buf225 = buf218; del buf218  # reuse
        buf223 = reinterpret_tensor(buf225, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_40], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_69.run(buf220, buf222, buf223, 399360, stream=stream0)
        buf224 = reinterpret_tensor(buf225, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [pow_37, mean_36, add_54, sqrt_36, hidden_states_90, hidden_states_91], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_70.run(buf220, buf222, buf224, 998400, stream=stream0)
        del buf223
        del buf224
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, arg89_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg89_1
        buf227 = buf221; del buf221  # reuse
        # Topologically Sorted Source Nodes: [x_88, pow_38, mean_37], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_60.run(buf226, arg90_1, buf227, 7800, 128, stream=stream0)
        buf228 = buf222; del buf222  # reuse
        # Topologically Sorted Source Nodes: [x_88, pow_38, mean_37], Original ATen: [aten.convolution, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf227, buf228, 1950, 4, stream=stream0)
        buf231 = buf225; del buf225  # reuse
        buf229 = reinterpret_tensor(buf231, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_41], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_62.run(buf226, arg90_1, buf228, buf229, 399360, stream=stream0)
        buf230 = reinterpret_tensor(buf231, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_88, pow_38, mean_37, add_55, sqrt_37, hidden_states_92, hidden_states_93], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_63.run(buf226, arg90_1, buf228, buf230, 998400, stream=stream0)
        del arg90_1
        del buf226
        del buf229
        del buf230
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf232 = extern_kernels.convolution(buf231, arg91_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (1, 512, 5, 26, 15), (998400, 1950, 390, 15, 1))
        del arg91_1
        buf233 = buf227; del buf227  # reuse
        # Topologically Sorted Source Nodes: [x_90, output_tensor_18, pow_39, mean_38], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_71.run(buf220, buf232, arg92_1, buf233, 7800, 128, stream=stream0)
        buf234 = buf228; del buf228  # reuse
        # Topologically Sorted Source Nodes: [x_90, output_tensor_18, pow_39, mean_38], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_61.run(buf233, buf234, 1950, 4, stream=stream0)
        del buf233
        buf237 = buf231; del buf231  # reuse
        buf235 = reinterpret_tensor(buf237, (1, 512, 2, 26, 15), (1397760, 2730, 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_42], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_72.run(buf220, buf232, arg92_1, buf234, buf235, 399360, stream=stream0)
        buf236 = reinterpret_tensor(buf237, (1, 512, 5, 26, 15), (1397760, 2730, 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_90, output_tensor_18, pow_39, mean_38, add_57, sqrt_38, sample, sample_1], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_73.run(buf220, buf232, arg92_1, buf234, buf236, 998400, stream=stream0)
        del arg92_1
        del buf220
        del buf232
        del buf234
        del buf235
        del buf236
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, arg93_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (1, 129, 5, 26, 15), (251550, 1950, 390, 15, 1))
        del arg93_1
        del buf237
        buf239 = empty_strided_cuda((1, 256, 5, 26, 15), (499200, 1950, 390, 15, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.cat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_75.run(buf238, arg94_1, buf239, 499200, stream=stream0)
        del arg94_1
        del buf238
    return (buf239, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 3, 33, 832, 480), (39536640, 13178880, 399360, 480, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((128, 48, 3, 3, 3), (1296, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((256, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((512, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((129, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((129, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
