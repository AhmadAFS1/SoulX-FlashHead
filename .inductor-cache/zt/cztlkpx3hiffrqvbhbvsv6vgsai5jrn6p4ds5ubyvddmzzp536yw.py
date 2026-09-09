# AOT ID: ['9_inference']
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/op/copqaj6rbztgblv7rt7lgixe4hyewpiivjqthhvuqbvec6bg7pea.py
# Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_1 => cat
#   x_2 => convolution
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg4_1, %arg5_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_0 = async_compile.triton('triton_poi_fused_cat_convolution_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_0(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 24960) % ks0)
    x0 = (xindex % 120)
    x1 = ((xindex // 120) % 208)
    x3 = xindex // ks1
    x4 = xindex
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (4*x0 + 480*((x3 % 4)) + 1920*x1 + ks2*(x3 // 16) + (((x3 // 4) % 4))), xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (4*x0 + 480*((x3 % 4)) + 1920*x1 + 399360*((-2) + x2) + ks2*(x3 // 16) + (((x3 // 4) % 4))), xmask & tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x4), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/5p/c5p5ulhgdxjk5zz5cokkbdrjb3l6jqzzoygffgzezeydiwj3bq6f.py
# Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean => mean
#   pow_1 => pow_1
#   x_1 => cat
#   x_2 => convolution
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg4_1, %arg5_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
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
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_pow_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_pow_1(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 24960*ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/2q/c2q56jx6hr25kptpdhkplc5k7jn6tuygwhgk4x7xx3o7mlmfbsez.py
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_2', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6389760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24960)
    x2 = xindex // 49920
    x3 = (xindex % 49920)
    tmp0 = tl.load(in_ptr0 + (x0 + 24960*ks0*x2), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 49920*x2 + 24960*ks0*x2), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/k7/ck7hb4cqqdy3sqaogqg3wegbtgcozwrh7jav6lk2fmojdxi75moe.py
# Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean, add, sqrt, hidden_states, hidden_states_1], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add => add_73
#   hidden_states => div
#   hidden_states_1 => convert_element_type, convert_element_type_1, mul_106, sigmoid
#   mean => mean
#   pow_1 => pow_1
#   sqrt => sqrt
#   x_1 => cat
#   x_2 => convolution
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg4_1, %arg5_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [1], True), kwargs = {})
#   %add_73 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean, 1e-08), kwargs = {})
#   %sqrt : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_73,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution, %sqrt), kwargs = {})
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type,), kwargs = {})
#   %mul_106 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type, %sigmoid), kwargs = {})
#   %convert_element_type_1 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_106, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 49920*x1 + 24960*ks1*x1), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/wq/cwqzpwmpm3zf7nah7lgmuswwazwnphec62sb2c7rrphmzm5ykxqh.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_2 => mean_2
#   output_tensor => add_195
#   pow_3 => pow_3
#   x_1 => cat
#   x_2 => convolution
#   x_6 => convolution_2
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg4_1, %arg5_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg8_1, %arg9_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_195 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_195, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [1], True), kwargs = {})
triton_red_fused_add_cat_convolution_mean_pow_4 = async_compile.triton('triton_red_fused_add_cat_convolution_mean_pow_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_convolution_mean_pow_4', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_convolution_mean_pow_4(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 24960*ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 24960*ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/en/cen7pcepp7rxczixw2dpdniq5j3flkj4fzcydoh3fecgwpcsbtfo.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2, add_3, sqrt_2, hidden_states_5, hidden_states_6], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_3 => add_210
#   hidden_states_5 => div_2
#   hidden_states_6 => convert_element_type_4
#   mean_2 => mean_2
#   output_tensor => add_195
#   pow_3 => pow_3
#   sqrt_2 => sqrt_2
#   x_1 => cat
#   x_2 => convolution
#   x_6 => convolution_2
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg4_1, %arg5_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg8_1, %arg9_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_195 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %pow_3 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_195, 2), kwargs = {})
#   %mean_2 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_3, [1], True), kwargs = {})
#   %add_210 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_2, 1e-08), kwargs = {})
#   %sqrt_2 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_210,), kwargs = {})
#   %div_2 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_195, %sqrt_2), kwargs = {})
#   %convert_element_type_4 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_2, torch.float32), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/hc/chc7lr5zsmaojbyooijlq36m2hgqtggdok3hprffngcrkhe2pdam.py
# Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_7 => cat_3
#   x_8 => convolution_3
# Graph fragment:
#   %cat_3 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_3, %convert_element_type_5], 2), kwargs = {})
#   %convolution_3 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_3, %arg10_1, %arg11_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_6 = async_compile.triton('triton_poi_fused_cat_convolution_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_6(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 24960) % ks0)
    x0 = (xindex % 24960)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 24960*ks2*x2), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = ks0
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 24960*((-2) + x1) + 24960*ks2*x2), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/qr/cqrudxoroujrk4ehqbeggowjfclwcbrwdztz2iho7vny77ocxdfo.py
# Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, x_10, output_tensor_1], Original ATen: [aten.cat, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor => add_195
#   output_tensor_1 => add_332
#   x_1 => cat
#   x_10 => convolution_4
#   x_2 => convolution
#   x_6 => convolution_2
# Graph fragment:
#   %cat : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat, %view_1], 2), kwargs = {})
#   %convolution : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat, %arg4_1, %arg5_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_2 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_2, %arg8_1, %arg9_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_195 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution, %convolution_2), kwargs = {})
#   %convolution_4 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_4, %arg12_1, %arg13_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_332 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_195, %convolution_4), kwargs = {})
triton_poi_fused_add_cat_convolution_7 = async_compile.triton('triton_poi_fused_add_cat_convolution_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_7', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/yg/cyg7ihpanaqpn2634kgztvf2eo3nocpioeegvhj5dti5zdhkduh2.py
# Topologically Sorted Source Nodes: [pow_5, mean_4], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_4 => mean_4
#   pow_5 => pow_5
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_332, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [1], True), kwargs = {})
triton_red_fused_mean_pow_8 = async_compile.triton('triton_red_fused_mean_pow_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_8', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_8(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 24960*ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/br/cbrsjhp3crq3em5ryiqo2ixrtpprgr7t2fxysw6eiiidx4qqy65j.py
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_9(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6389760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24960)
    x2 = xindex // 49920
    x3 = (xindex % 49920)
    tmp0 = tl.load(in_ptr0 + (x0 + 24960*ks0*x2), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 49920*x2 + 24960*ks0*x2), tmp12, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/qh/cqhmeaaer6qb4o73jrk5r2b7dycsic37qc7oocai6xxzxjawxkwu.py
# Topologically Sorted Source Nodes: [pow_5, mean_4, add_6, sqrt_4, hidden_states_10, hidden_states_11], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_6 => add_347
#   hidden_states_10 => div_4
#   hidden_states_11 => convert_element_type_8, convert_element_type_9, mul_322, sigmoid_4
#   mean_4 => mean_4
#   pow_5 => pow_5
#   sqrt_4 => sqrt_4
# Graph fragment:
#   %pow_5 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_332, 2), kwargs = {})
#   %mean_4 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_5, [1], True), kwargs = {})
#   %add_347 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_4, 1e-08), kwargs = {})
#   %sqrt_4 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_347,), kwargs = {})
#   %div_4 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_332, %sqrt_4), kwargs = {})
#   %convert_element_type_8 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_4, torch.float32), kwargs = {})
#   %sigmoid_4 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_8,), kwargs = {})
#   %mul_322 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_8, %sigmoid_4), kwargs = {})
#   %convert_element_type_9 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_322, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_10 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_10(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 49920*x1 + 24960*ks1*x1), tmp12, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/mv/cmvsbymfav5r6qaq6p4wfaevfuclrzfqgkyvaqas2qyccewwoxo7.py
# Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_6 => mean_6
#   output_tensor_2 => add_469
#   pow_7 => pow_7
#   x_14 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_6, %arg16_1, %arg17_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_469 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_332, %convolution_6), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_469, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_11 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 262144, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_11', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_11(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 24960*ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 24960*ks0*r0_1), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/hc/chc2cf6ftbupwcdtjubdsh7xbp5bysoqk6itjgwxtxjce5vah6yx.py
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_12', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6389760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 24960)
    x2 = xindex // 49920
    x3 = (xindex % 49920)
    tmp0 = tl.load(in_ptr0 + (x0 + 24960*ks0*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 24960*ks0*x2), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 49920*x2 + 24960*ks0*x2), tmp16, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/r6/cr6hojbmecsv3aapgpzdvuj7wqvbjix27pfrcobmi3forvd5yzlz.py
# Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6, add_9, sqrt_6, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_9 => add_484
#   hidden_states_15 => div_6
#   hidden_states_16 => convert_element_type_12, convert_element_type_13, mul_430, sigmoid_6
#   mean_6 => mean_6
#   output_tensor_2 => add_469
#   pow_7 => pow_7
#   sqrt_6 => sqrt_6
#   x_14 => convolution_6
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_6, %arg16_1, %arg17_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_469 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_332, %convolution_6), kwargs = {})
#   %pow_7 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_469, 2), kwargs = {})
#   %mean_6 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_7, [1], True), kwargs = {})
#   %add_484 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_6, 1e-08), kwargs = {})
#   %sqrt_6 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_484,), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_469, %sqrt_6), kwargs = {})
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_6, torch.float32), kwargs = {})
#   %sigmoid_6 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_12,), kwargs = {})
#   %mul_430 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %sigmoid_6), kwargs = {})
#   %convert_element_type_13 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_430, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 49920*x1 + 24960*ks1*x1), tmp16, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/or/cor5g2b7je5x5w46gehqqekne3svcc2whdymcoq562vcgcdse4vr.py
# Topologically Sorted Source Nodes: [x_14, output_tensor_2, x_18, output_tensor_3], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_2 => add_469
#   output_tensor_3 => add_606
#   x_14 => convolution_6
#   x_18 => convolution_8
# Graph fragment:
#   %convolution_6 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_6, %arg16_1, %arg17_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_469 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_332, %convolution_6), kwargs = {})
#   %convolution_8 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_8, %arg20_1, %arg21_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_606 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_469, %convolution_8), kwargs = {})
triton_poi_fused_add_convolution_14 = async_compile.triton('triton_poi_fused_add_convolution_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 33554432}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_14', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/he/chefz7tesuoazwflkj4c6nrwrxyq3e4dqusax77q4smnyplnflyq.py
# Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_19 => cat_9
#   x_20 => convolution_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_9, %add_606], 2), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_9, %arg22_1, %arg23_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_15 = async_compile.triton('triton_poi_fused_cat_convolution_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 67108864}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_15', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_15(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 24960) % ks0)
    x0 = (xindex % 24960)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 24960*ks2*x2), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 24960*((-2) + x1) + 24960*ks2*x2), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/gy/cgyoptualxtadx4rkdjffb55hh5m3lfr27gpp7wfvkszsr4s5rn7.py
# Topologically Sorted Source Nodes: [x_19, x_20, x_26, pow_9, mean_8], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_8 => mean_8
#   pow_9 => pow_9
#   x_19 => cat_9
#   x_20 => convolution_9
#   x_26 => clone_6, convert_element_type_20, var_mean
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_9, %add_606], 2), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_9, %arg22_1, %arg23_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %clone_6 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_1,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_20 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_6, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_20, [4]), kwargs = {correction: 0, keepdim: True})
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
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 6240*r0_1 + 6240*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  2))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/qd/cqd7qtyk4t7fxbqpnys2txhy5f46n3npviv5rtvrs4rpijsrnmul.py
# Topologically Sorted Source Nodes: [x_19, x_20, input_tensor, pow_9, mean_8, add_12, sqrt_8, hidden_states_20, hidden_states_21], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_12 => add_656
#   hidden_states_20 => div_8
#   hidden_states_21 => convert_element_type_16, convert_element_type_17, mul_563, sigmoid_8
#   input_tensor => convolution_12
#   mean_8 => mean_8
#   pow_9 => pow_9
#   sqrt_8 => sqrt_8
#   x_19 => cat_9
#   x_20 => convolution_9
# Graph fragment:
#   %cat_9 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_9, %add_606], 2), kwargs = {})
#   %convolution_9 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_9, %arg22_1, %arg23_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg30_1, %arg31_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_9 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_9, 2), kwargs = {})
#   %mean_8 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_9, [1], True), kwargs = {})
#   %add_656 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_8, 1e-08), kwargs = {})
#   %sqrt_8 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_656,), kwargs = {})
#   %div_8 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_9, %sqrt_8), kwargs = {})
#   %convert_element_type_16 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_8, torch.float32), kwargs = {})
#   %sigmoid_8 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_16,), kwargs = {})
#   %mul_563 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_16, %sigmoid_8), kwargs = {})
#   %convert_element_type_17 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_563, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_17 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 128, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_17', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 128.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = (tmp20 / tmp7)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = 1e-08
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = (tmp2 / tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.sigmoid(tmp27)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 128*x1), tmp19, xmask & ymask)
    tl.store(out_ptr1 + (x1 + 18720*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), tmp30, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/65/c65rt4eov7m4mdoii3siwuxvbej6rzvmaujqs4zgq4zb73kxbpwd.py
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_18', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_18(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6240)
    x2 = xindex // 12480
    x3 = (xindex % 12480)
    tmp0 = tl.load(in_ptr0 + (x0 + 6240*x2 + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks0,  2))), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 18720*x2 + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks0,  2))), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/iv/civxn4pmj6x2oepzqsq2bgr6oou5ft2w5icln3qwl5fe4g4s2lw6.py
# Topologically Sorted Source Nodes: [x_22, pow_10, mean_9], Original ATen: [aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_9 => mean_9
#   pow_10 => pow_10
#   x_22 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_10, %arg24_1, %arg25_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_10, 2), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [1], True), kwargs = {})
triton_red_fused_convolution_mean_pow_19 = async_compile.triton('triton_red_fused_convolution_mean_pow_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_pow_19', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_pow_19(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (x0 + 6240*r0_1 + 6240*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  2))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/tl/ctlzerrvlb7jbb5pavw7tifu3ah5xeb2ak7f7stdkqriz37ijcn4.py
# Topologically Sorted Source Nodes: [first_frame_pad_11], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_11 => repeat_11
# Graph fragment:
#   %repeat_11 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_58, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_20 = async_compile.triton('triton_poi_fused_repeat_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_20', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_20(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3194880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 6240)
    x2 = xindex // 12480
    x3 = (xindex % 12480)
    tmp0 = tl.load(in_ptr0 + (x0 + 6240*x2 + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks0,  2))), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 18720*x2 + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks0,  2))), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/kn/ckn2uwpt7272on2pdc4zrtjmuew6ybfod7xfakytj6qmoo6ujcwx.py
# Topologically Sorted Source Nodes: [x_22, pow_10, mean_9, add_13, sqrt_9, hidden_states_22, hidden_states_23], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_13 => add_719
#   hidden_states_22 => div_9
#   hidden_states_23 => convert_element_type_18, convert_element_type_19, mul_613, sigmoid_9
#   mean_9 => mean_9
#   pow_10 => pow_10
#   sqrt_9 => sqrt_9
#   x_22 => convolution_10
# Graph fragment:
#   %convolution_10 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_10, %arg24_1, %arg25_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_10 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_10, 2), kwargs = {})
#   %mean_9 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_10, [1], True), kwargs = {})
#   %add_719 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_9, 1e-08), kwargs = {})
#   %sqrt_9 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_719,), kwargs = {})
#   %div_9 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_10, %sqrt_9), kwargs = {})
#   %convert_element_type_18 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_9, torch.float32), kwargs = {})
#   %sigmoid_9 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_18,), kwargs = {})
#   %mul_613 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_18, %sigmoid_9), kwargs = {})
#   %convert_element_type_19 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_613, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 18720*x1 + 6240*x1*(triton_helpers.div_floor_integer((-1) + ks1,  2))), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ew/cewmt6sbabtenxfhambx73kf5hxfink5oraf3zryip32fsd3anxq.py
# Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   input_tensor => convolution_12
#   mean_10 => mean_10
#   output_tensor_4 => add_808
#   pow_11 => pow_11
#   x_24 => convolution_11
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg30_1, %arg31_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg26_1, %arg27_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_808 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_808, 2), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_11, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_22 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_22', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 256*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 6240*r0_1 + 6240*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  2))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/jr/cjrq2cv373phrpap42o2wt3sr2ddozsslevjnfq64nhu6mj3hcvg.py
# Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10, add_15, sqrt_10, hidden_states_25, hidden_states_26], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_15 => add_821
#   hidden_states_25 => div_10
#   hidden_states_26 => convert_element_type_22
#   input_tensor => convolution_12
#   mean_10 => mean_10
#   output_tensor_4 => add_808
#   pow_11 => pow_11
#   sqrt_10 => sqrt_10
#   x_24 => convolution_11
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg30_1, %arg31_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg26_1, %arg27_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_808 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %pow_11 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_808, 2), kwargs = {})
#   %mean_10 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_11, [1], True), kwargs = {})
#   %add_821 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_10, 1e-08), kwargs = {})
#   %sqrt_10 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_821,), kwargs = {})
#   %div_10 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_808, %sqrt_10), kwargs = {})
#   %convert_element_type_22 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_10, torch.float32), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_23 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_23', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_23(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/6d/c6dn23ivrbmuowsev7dowqthuoqqu7aknedziuwkqij3d2dgtdru.py
# Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_28 => cat_12
#   x_29 => convolution_13
# Graph fragment:
#   %cat_12 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_12, %convert_element_type_23], 2), kwargs = {})
#   %convolution_13 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_12, %arg32_1, %arg33_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_24 = async_compile.triton('triton_poi_fused_cat_convolution_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_24', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_24(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6240) % ks0)
    x0 = (xindex % 6240)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 6240*x2 + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks2,  2))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = ks0
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 6240*x2 + 6240*((-2) + x1) + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks2,  2))), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/rr/crr7q5olfxqe6oxggea7o3augq4qjhlgquiomfvm2h6p7rtyohrf.py
# Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, x_31, output_tensor_5], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_tensor => convolution_12
#   output_tensor_4 => add_808
#   output_tensor_5 => add_924
#   x_24 => convolution_11
#   x_31 => convolution_14
# Graph fragment:
#   %convolution_12 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_2, %arg30_1, %arg31_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_11 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_11, %arg26_1, %arg27_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_808 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_12, %convolution_11), kwargs = {})
#   %convolution_14 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_13, %arg34_1, %arg35_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_924 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_808, %convolution_14), kwargs = {})
triton_poi_fused_add_convolution_25 = async_compile.triton('triton_poi_fused_add_convolution_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_25', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_25(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/l6/cl6onv4cvflh6dqtbahh5p72zcnmnfwfvijlfbt3uzpdaegzbs2n.py
# Topologically Sorted Source Nodes: [pow_13, mean_12], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_12 => mean_12
#   pow_13 => pow_13
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_924, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_13, [1], True), kwargs = {})
triton_red_fused_mean_pow_26 = async_compile.triton('triton_red_fused_mean_pow_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_26', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_26(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + 6240*r0_1 + 6240*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  2))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/km/ckmp2spiliihs7bo6t2k7yfp37mkyinukn5zgeguozabn6nxxw4q.py
# Topologically Sorted Source Nodes: [first_frame_pad_14], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_14 => repeat_14
# Graph fragment:
#   %repeat_14 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_73, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_27 = async_compile.triton('triton_poi_fused_repeat_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_27', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_27(in_ptr0, in_ptr1, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/eg/cegsdwwxu7kkrjitysajmn7q7bm3m545thh6jgnqrx55nvkxaukh.py
# Topologically Sorted Source Nodes: [pow_13, mean_12, add_18, sqrt_12, hidden_states_30, hidden_states_31], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_18 => add_937
#   hidden_states_30 => div_12
#   hidden_states_31 => convert_element_type_26, convert_element_type_27, mul_785, sigmoid_12
#   mean_12 => mean_12
#   pow_13 => pow_13
#   sqrt_12 => sqrt_12
# Graph fragment:
#   %pow_13 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_924, 2), kwargs = {})
#   %mean_12 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_13, [1], True), kwargs = {})
#   %add_937 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_12, 1e-08), kwargs = {})
#   %sqrt_12 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_937,), kwargs = {})
#   %div_12 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_924, %sqrt_12), kwargs = {})
#   %convert_element_type_26 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_12, torch.float32), kwargs = {})
#   %sigmoid_12 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_26,), kwargs = {})
#   %mul_785 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_26, %sigmoid_12), kwargs = {})
#   %convert_element_type_27 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_785, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_28 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_28', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_28(in_ptr0, in_ptr1, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + 256*x1), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/7u/c7ugsxknne2gewm3f7qqyv7cjibngm5nuo7gf6dftwqpxhem2csd.py
# Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_33 => convolution_15
# Graph fragment:
#   %convolution_15 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_14, %arg36_1, %arg37_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_29 = async_compile.triton('triton_poi_fused_convolution_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 65536}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_29', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_29(in_ptr0, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 256*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 18720*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/3s/c3sro4emqh62fyskxibsgsplg4qqkrucchh2jfk65ykuviuf4nxa.py
# Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_14 => mean_14
#   output_tensor_6 => add_1040
#   pow_15 => pow_15
#   x_35 => convolution_16
# Graph fragment:
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %arg38_1, %arg39_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1040 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_924, %convolution_16), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1040, 2), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_15, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_30 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_30', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 256
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
        tmp0 = tl.load(in_ptr0 + (x0 + 6240*r0_1 + 6240*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  2))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 6240*r0_1 + 6240*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  2))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/3v/c3v4s2ym2c5o3wef2vuxk5vwf4oqnn753hhu2x42u3bhru76jyyr.py
# Topologically Sorted Source Nodes: [first_frame_pad_16], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_16 => repeat_16
# Graph fragment:
#   %repeat_16 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_83, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_31 = async_compile.triton('triton_poi_fused_repeat_31', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_31', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/l6/cl6xhukdxo2dt5tqszzcvkpdkt7jslg6l7hizo52sbwzx2s6ttvy.py
# Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14, add_21, sqrt_14, hidden_states_35, hidden_states_36], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_21 => add_1053
#   hidden_states_35 => div_14
#   hidden_states_36 => convert_element_type_30, convert_element_type_31, mul_881, sigmoid_14
#   mean_14 => mean_14
#   output_tensor_6 => add_1040
#   pow_15 => pow_15
#   sqrt_14 => sqrt_14
#   x_35 => convolution_16
# Graph fragment:
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %arg38_1, %arg39_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1040 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_924, %convolution_16), kwargs = {})
#   %pow_15 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1040, 2), kwargs = {})
#   %mean_14 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_15, [1], True), kwargs = {})
#   %add_1053 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_14, 1e-08), kwargs = {})
#   %sqrt_14 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1053,), kwargs = {})
#   %div_14 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1040, %sqrt_14), kwargs = {})
#   %convert_element_type_30 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_14, torch.float32), kwargs = {})
#   %sigmoid_14 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_30,), kwargs = {})
#   %mul_881 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_30, %sigmoid_14), kwargs = {})
#   %convert_element_type_31 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_881, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_32 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_32', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 32768}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_32', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + 6240*y0 + 6240*y0*(triton_helpers.div_floor_integer((-1) + ks0,  2))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + 256*x1), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ly/clytozqryhdqwrnwgfhtnug43pplvap2nvujbzusresic3csazwb.py
# Topologically Sorted Source Nodes: [x_35, output_tensor_6, x_39, output_tensor_7], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_6 => add_1040
#   output_tensor_7 => add_1156
#   x_35 => convolution_16
#   x_39 => convolution_18
# Graph fragment:
#   %convolution_16 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_15, %arg38_1, %arg39_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1040 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_924, %convolution_16), kwargs = {})
#   %convolution_18 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_17, %arg42_1, %arg43_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1156 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1040, %convolution_18), kwargs = {})
triton_poi_fused_add_convolution_33 = async_compile.triton('triton_poi_fused_add_convolution_33', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8388608}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_33', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/c4/cc4dye7t46w3cmz6qni5q5f53j5hdzu53wkxtyry2t5owfnh7qta.py
# Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_40 => cat_18
#   x_41 => convolution_19
# Graph fragment:
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_18, %add_1156], 2), kwargs = {})
#   %convolution_19 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_18, %arg44_1, %arg45_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_34 = async_compile.triton('triton_poi_fused_cat_convolution_34', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16777216}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_34', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_34(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 6240) % ks0)
    x0 = (xindex % 6240)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 6240*x2 + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks2,  2))), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 6240*x2 + 6240*((-2) + x1) + 6240*x2*(triton_helpers.div_floor_integer((-1) + ks2,  2))), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/lc/clc4jj2xo4w3ecw3khyj5ekznzowe2iwuspxjdqtdy2wfk3fig42.py
# Topologically Sorted Source Nodes: [x_40, x_41, x_47, pow_17, mean_16], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_16 => mean_16
#   pow_17 => pow_17
#   x_40 => cat_18
#   x_41 => convolution_19
#   x_47 => clone_11, convert_element_type_38, var_mean_1
# Graph fragment:
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_18, %add_1156], 2), kwargs = {})
#   %convolution_19 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_18, %arg44_1, %arg45_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %clone_11 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_3,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_38 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clone_11, torch.float32), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_38, [4]), kwargs = {correction: 0, keepdim: True})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_19, 2), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [1], True), kwargs = {})
triton_red_fused_cat_convolution_mean_native_layer_norm_pow_35 = async_compile.triton('triton_red_fused_cat_convolution_mean_native_layer_norm_pow_35', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_native_layer_norm_pow_35', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 3, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_native_layer_norm_pow_35(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_1 + 1560*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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
    tl.store(out_ptr0 + (x0), tmp5, xmask)
    tl.store(out_ptr1 + (x0), tmp6, xmask)
    tl.store(out_ptr2 + (x0), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/tq/ctqirxf5yysuul6jquxz7m2k4mpccyoutdwpnlv25636vfj3f7aq.py
# Topologically Sorted Source Nodes: [x_40, x_41, input_tensor_1, pow_17, mean_16, add_24, sqrt_16, hidden_states_40, hidden_states_41], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_24 => add_1188
#   hidden_states_40 => div_16
#   hidden_states_41 => convert_element_type_34, convert_element_type_35, mul_996, sigmoid_16
#   input_tensor_1 => convolution_22
#   mean_16 => mean_16
#   pow_17 => pow_17
#   sqrt_16 => sqrt_16
#   x_40 => cat_18
#   x_41 => convolution_19
# Graph fragment:
#   %cat_18 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_18, %add_1156], 2), kwargs = {})
#   %convolution_19 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_18, %arg44_1, %arg45_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg52_1, %arg53_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_17 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_19, 2), kwargs = {})
#   %mean_16 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_17, [1], True), kwargs = {})
#   %add_1188 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_16, 1e-08), kwargs = {})
#   %sqrt_16 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1188,), kwargs = {})
#   %div_16 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_19, %sqrt_16), kwargs = {})
#   %convert_element_type_34 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_16, torch.float32), kwargs = {})
#   %sigmoid_16 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_34,), kwargs = {})
#   %mul_996 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_34, %sigmoid_16), kwargs = {})
#   %convert_element_type_35 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_996, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_36 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_36', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 256, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*bf16', 'in_ptr5': '*bf16', 'in_ptr6': '*fp32', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_36', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 - tmp4
    tmp7 = 256.0
    tmp8 = (tmp6 / tmp7)
    tmp9 = 1e-06
    tmp10 = tmp8 + tmp9
    tmp11 = libdevice.rsqrt(tmp10)
    tmp12 = tmp5 * tmp11
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp12 * tmp14
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = (tmp20 / tmp7)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = 1e-08
    tmp24 = tmp22 + tmp23
    tmp25 = libdevice.sqrt(tmp24)
    tmp26 = (tmp2 / tmp25)
    tmp27 = tmp26.to(tl.float32)
    tmp28 = tl.sigmoid(tmp27)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tl.store(out_ptr0 + (y0 + 256*x1), tmp19, xmask & ymask)
    tl.store(out_ptr1 + (x1 + 4680*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), tmp30, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/qq/cqqiznl2t4ysjkd2qzigjlwm6mfsrqbpi3tpuymymgvwmqxfuvjq.py
# Topologically Sorted Source Nodes: [first_frame_pad_19], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_19 => repeat_19
# Graph fragment:
#   %repeat_19 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_98, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_37 = async_compile.triton('triton_poi_fused_repeat_37', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_37', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_37(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 798720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1560)
    x2 = xindex // 3120
    x3 = (xindex % 3120)
    tmp0 = tl.load(in_ptr0 + (x0 + 1560*x2 + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks0,  4))), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 4680*x2 + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks0,  4))), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/iv/civuzawenpbxclus5udoz4g44bm4cef5zitpdqfsjoeafuwbu626.py
# Topologically Sorted Source Nodes: [x_43, pow_18, mean_17], Original ATen: [aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_17 => mean_17
#   pow_18 => pow_18
#   x_43 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_19, %arg46_1, %arg47_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_18 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_20, 2), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_18, [1], True), kwargs = {})
triton_red_fused_convolution_mean_pow_38 = async_compile.triton('triton_red_fused_convolution_mean_pow_38', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_pow_38', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_convolution_mean_pow_38(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_1 + 1560*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/gm/cgmrbkxnh5l6c3v5st2ufo55jpxwot3j4c4fsusaq7a7ltbogpi5.py
# Topologically Sorted Source Nodes: [first_frame_pad_20], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_20 => repeat_20
# Graph fragment:
#   %repeat_20 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_103, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_39 = async_compile.triton('triton_poi_fused_repeat_39', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_39', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_39(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1597440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1560)
    x2 = xindex // 3120
    x3 = (xindex % 3120)
    tmp0 = tl.load(in_ptr0 + (x0 + 1560*x2 + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks0,  4))), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 4680*x2 + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks0,  4))), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/cr/ccrusxi2q4ddaynxgcqkpg4rqyykjszfi534rds3tr56lvzl4o23.py
# Topologically Sorted Source Nodes: [x_43, pow_18, mean_17, add_25, sqrt_17, hidden_states_42, hidden_states_43], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_25 => add_1251
#   hidden_states_42 => div_17
#   hidden_states_43 => convert_element_type_36, convert_element_type_37, mul_1046, sigmoid_17
#   mean_17 => mean_17
#   pow_18 => pow_18
#   sqrt_17 => sqrt_17
#   x_43 => convolution_20
# Graph fragment:
#   %convolution_20 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_19, %arg46_1, %arg47_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_18 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_20, 2), kwargs = {})
#   %mean_17 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_18, [1], True), kwargs = {})
#   %add_1251 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_17, 1e-08), kwargs = {})
#   %sqrt_17 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1251,), kwargs = {})
#   %div_17 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_20, %sqrt_17), kwargs = {})
#   %convert_element_type_36 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_17, torch.float32), kwargs = {})
#   %sigmoid_17 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_36,), kwargs = {})
#   %mul_1046 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_36, %sigmoid_17), kwargs = {})
#   %convert_element_type_37 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1046, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 4680*x1 + 1560*x1*(triton_helpers.div_floor_integer((-1) + ks1,  4))), tmp14, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/wb/cwbpsetjxi5avmfcxbgwjdogiup4yxnsomazxmllhprw5ephlpnd.py
# Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   input_tensor_1 => convolution_22
#   mean_18 => mean_18
#   output_tensor_8 => add_1340
#   pow_19 => pow_19
#   x_45 => convolution_21
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg52_1, %arg53_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_20, %arg48_1, %arg49_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1340 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1340, 2), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_19, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_41 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_41', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_41', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_41(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 512
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
        tmp0 = tl.load(in_ptr0 + (r0_1 + 512*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 1560*r0_1 + 1560*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/rn/crnewlrabpq4keikzfmyjudwq7ascdlrb3dybtzeavx6xxumc534.py
# Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18, add_27, sqrt_18, hidden_states_45, hidden_states_46], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_27 => add_1353
#   hidden_states_45 => div_18
#   hidden_states_46 => convert_element_type_40
#   input_tensor_1 => convolution_22
#   mean_18 => mean_18
#   output_tensor_8 => add_1340
#   pow_19 => pow_19
#   sqrt_18 => sqrt_18
#   x_45 => convolution_21
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg52_1, %arg53_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_20, %arg48_1, %arg49_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1340 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %pow_19 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1340, 2), kwargs = {})
#   %mean_18 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_19, [1], True), kwargs = {})
#   %add_1353 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_18, 1e-08), kwargs = {})
#   %sqrt_18 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1353,), kwargs = {})
#   %div_18 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1340, %sqrt_18), kwargs = {})
#   %convert_element_type_40 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_18, torch.float32), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_42 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_42', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_42', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), tmp15, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/4e/c4efwtmt2rsebmm74dowlzv6uncoa52c4vwfl2d6qhcremgqwsef.py
# Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_49 => cat_21
#   x_50 => convolution_23
# Graph fragment:
#   %cat_21 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_21, %convert_element_type_41], 2), kwargs = {})
#   %convolution_23 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_21, %arg54_1, %arg55_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_43 = async_compile.triton('triton_poi_fused_cat_convolution_43', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_43', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_43(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1560) % ks0)
    x0 = (xindex % 1560)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1560*x2 + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = ks0
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 1560*x2 + 1560*((-2) + x1) + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/tq/ctqocwf64zsskbuhsbnu7khngkxaeuxzjqyuupifyq4whxhtxfrl.py
# Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, x_52, output_tensor_9], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   input_tensor_1 => convolution_22
#   output_tensor_8 => add_1340
#   output_tensor_9 => add_1456
#   x_45 => convolution_21
#   x_52 => convolution_24
# Graph fragment:
#   %convolution_22 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%permute_4, %arg52_1, %arg53_1, [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_21 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_20, %arg48_1, %arg49_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1340 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_22, %convolution_21), kwargs = {})
#   %convolution_24 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_22, %arg56_1, %arg57_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1456 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1340, %convolution_24), kwargs = {})
triton_poi_fused_add_convolution_44 = async_compile.triton('triton_poi_fused_add_convolution_44', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_44', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_44(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_out_ptr0 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), tmp10, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/of/cofe2asfw5sqsgmfco62pprdrohofkvwotodg4ovah3tpinyqcdz.py
# Topologically Sorted Source Nodes: [pow_21, mean_20], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_20 => mean_20
#   pow_21 => pow_21
# Graph fragment:
#   %pow_21 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1456, 2), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_21, [1], True), kwargs = {})
triton_red_fused_mean_pow_45 = async_compile.triton('triton_red_fused_mean_pow_45', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_45', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_45(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_1 + 1560*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/6c/c6ccvdxemq7an6cchxrpu2i5hj4o2zisxongo2ztrvzflr7bm6w3.py
# Topologically Sorted Source Nodes: [first_frame_pad_23], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_23 => repeat_23
# Graph fragment:
#   %repeat_23 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_118, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_46 = async_compile.triton('triton_poi_fused_repeat_46', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 1024, 'x': 2048}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_46', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_46(in_ptr0, in_ptr1, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask, eviction_policy='evict_last').to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ji/cji6yp47jpkdryfwomr4eeqkbckaybdnwoth27kvvkb23ufs2lhi.py
# Topologically Sorted Source Nodes: [pow_21, mean_20, add_30, sqrt_20, hidden_states_50, hidden_states_51], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_30 => add_1469
#   hidden_states_50 => div_20
#   hidden_states_51 => convert_element_type_44, convert_element_type_45, mul_1218, sigmoid_20
#   mean_20 => mean_20
#   pow_21 => pow_21
#   sqrt_20 => sqrt_20
# Graph fragment:
#   %pow_21 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1456, 2), kwargs = {})
#   %mean_20 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_21, [1], True), kwargs = {})
#   %add_1469 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_20, 1e-08), kwargs = {})
#   %sqrt_20 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1469,), kwargs = {})
#   %div_20 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1456, %sqrt_20), kwargs = {})
#   %convert_element_type_44 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_20, torch.float32), kwargs = {})
#   %sigmoid_20 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_44,), kwargs = {})
#   %mul_1218 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_44, %sigmoid_20), kwargs = {})
#   %convert_element_type_45 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1218, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_47 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_47', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_47', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_47(in_ptr0, in_ptr1, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + 512*x1), tmp12, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/zi/cziyarir6qzy2fdgpgno3s4e4a3c73gx6popsnmoytk27neja6en.py
# Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
# Source node to ATen node mapping:
#   x_54 => convolution_25
# Graph fragment:
#   %convolution_25 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_23, %arg58_1, %arg59_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_convolution_48 = async_compile.triton('triton_poi_fused_convolution_48', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_48', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_convolution_48(in_ptr0, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 512*x1), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + 4680*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), tmp0, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/oc/coclhah6774z54jsk5zldt5vkwu4ckj56rhvyhrub3qyx3h6r5aq.py
# Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_22 => mean_22
#   output_tensor_10 => add_1572
#   pow_23 => pow_23
#   x_56 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_24, %arg60_1, %arg61_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1572 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1456, %convolution_26), kwargs = {})
#   %pow_23 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1572, 2), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_23, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_49 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_49', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 512},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_49', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_49(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 512
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
        tmp0 = tl.load(in_ptr0 + (x0 + 1560*r0_1 + 1560*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 1560*r0_1 + 1560*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  4))), r0_mask & xmask, eviction_policy='evict_first', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/nz/cnzrvubqbm3hqut7vponoh44eulmlgru4zlzftgixngmrb7dsuki.py
# Topologically Sorted Source Nodes: [first_frame_pad_25], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_25 => repeat_25
# Graph fragment:
#   %repeat_25 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_128, [1, 1, 2, 1, 1]), kwargs = {})
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_50', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_50(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x2 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask, eviction_policy='evict_last').to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/pt/cptnet4ws67xjbwnlnrjtcx34tircxwqj2bsf6runhgc5avl7kav.py
# Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22, add_33, sqrt_22, hidden_states_55, hidden_states_56], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_33 => add_1585
#   hidden_states_55 => div_22
#   hidden_states_56 => convert_element_type_48, convert_element_type_49, mul_1314, sigmoid_22
#   mean_22 => mean_22
#   output_tensor_10 => add_1572
#   pow_23 => pow_23
#   sqrt_22 => sqrt_22
#   x_56 => convolution_26
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_24, %arg60_1, %arg61_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1572 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1456, %convolution_26), kwargs = {})
#   %pow_23 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1572, 2), kwargs = {})
#   %mean_22 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_23, [1], True), kwargs = {})
#   %add_1585 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_22, 1e-08), kwargs = {})
#   %sqrt_22 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1585,), kwargs = {})
#   %div_22 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1572, %sqrt_22), kwargs = {})
#   %convert_element_type_48 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_22, torch.float32), kwargs = {})
#   %sigmoid_22 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_48,), kwargs = {})
#   %mul_1314 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_48, %sigmoid_22), kwargs = {})
#   %convert_element_type_49 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1314, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_51 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_51', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 8192}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_51', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_51(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1 + 1560*y0 + 1560*y0*(triton_helpers.div_floor_integer((-1) + ks0,  4))), xmask & ymask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + 512*x1), tmp16, xmask & ymask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/eo/ceoqxlghnjdne7gaplffmkodkiyrb5ejw3udeltua6qz2jn3xlma.py
# Topologically Sorted Source Nodes: [x_56, output_tensor_10, x_60, output_tensor_11], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_10 => add_1572
#   output_tensor_11 => add_1688
#   x_56 => convolution_26
#   x_60 => convolution_28
# Graph fragment:
#   %convolution_26 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_24, %arg60_1, %arg61_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1572 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1456, %convolution_26), kwargs = {})
#   %convolution_28 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_26, %arg64_1, %arg65_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1688 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1572, %convolution_28), kwargs = {})
triton_poi_fused_add_convolution_52 = async_compile.triton('triton_poi_fused_add_convolution_52', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_52', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_52(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/hk/chkj6vvv6ickna5lvxqqcdtd7vnlxmac7vv3xzpqstgefpdjamll.py
# Topologically Sorted Source Nodes: [x_61, x_62], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_61 => cat_27
#   x_62 => convolution_29
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_1688], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg66_1, %arg67_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_53 = async_compile.triton('triton_poi_fused_cat_convolution_53', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_53', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_53(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x1 = ((xindex // 1560) % ks0)
    x0 = (xindex % 1560)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 1560*x2 + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tmp0 >= tmp3
    tmp7 = ks0
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr0 + (x0 + 1560*x2 + 1560*((-2) + x1) + 1560*x2*(triton_helpers.div_floor_integer((-1) + ks2,  4))), tmp6, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tl.store(out_ptr0 + (x3), tmp10, None)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/gq/cgqagcfgk2usc7cjg5m55gepgahgfa2jdemtpwm3wzj4jeojum6p.py
# Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_24 => mean_24
#   pow_25 => pow_25
#   x_61 => cat_27
#   x_62 => convolution_29
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_1688], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg66_1, %arg67_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_29, 2), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_25, [1], True), kwargs = {})
triton_red_fused_cat_convolution_mean_pow_54 = async_compile.triton('triton_red_fused_cat_convolution_mean_pow_54', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_cat_convolution_mean_pow_54', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_cat_convolution_mean_pow_54(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 390*r0_2 + 49920*x1 + 390*r0_2*(triton_helpers.div_floor_integer((-1) + ks1,  8)) + 49920*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2 * tmp2
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/fu/cfuenhzxow5656gdmiht3zdq6xy47q6wuloigpw3mxpkq6gscilc.py
# Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_24 => mean_24
#   pow_25 => pow_25
#   x_61 => cat_27
#   x_62 => convolution_29
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_1688], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg66_1, %arg67_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_29, 2), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_25, [1], True), kwargs = {})
triton_per_fused_cat_convolution_mean_pow_55 = async_compile.triton('triton_per_fused_cat_convolution_mean_pow_55', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 1024, 'r0_': 4},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_convolution_mean_pow_55', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused_cat_convolution_mean_pow_55(in_ptr0, out_ptr0, ks0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_ptr0 + (x0 + 390*r0_1 + 390*r0_1*(triton_helpers.div_floor_integer((-1) + ks0,  8))), xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/kp/ckpvy2y3i6agbsy6dgsvsgkom2iugiglrivoaqslrw357t5oi6yu.py
# Topologically Sorted Source Nodes: [first_frame_pad_28], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_28 => repeat_28
# Graph fragment:
#   %repeat_28 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_143, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_56 = async_compile.triton('triton_poi_fused_repeat_56', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_56', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_56(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 399360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 390)
    x2 = xindex // 780
    x3 = (xindex % 780)
    tmp0 = tl.load(in_ptr0 + (x0 + 390*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks0,  8))), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 1170*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks0,  8))), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/2t/c2t5l6sqwjcftsz67uv5yyrbqt4jv6wrb2cbod6e5v2cwvtrgjpb.py
# Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24, add_36, sqrt_24, hidden_states_60, hidden_states_61], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_36 => add_1720
#   hidden_states_60 => div_24
#   hidden_states_61 => convert_element_type_52, convert_element_type_53, mul_1429, sigmoid_24
#   mean_24 => mean_24
#   pow_25 => pow_25
#   sqrt_24 => sqrt_24
#   x_61 => cat_27
#   x_62 => convolution_29
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_1688], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg66_1, %arg67_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %pow_25 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convolution_29, 2), kwargs = {})
#   %mean_24 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_25, [1], True), kwargs = {})
#   %add_1720 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_24, 1e-08), kwargs = {})
#   %sqrt_24 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1720,), kwargs = {})
#   %div_24 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%convolution_29, %sqrt_24), kwargs = {})
#   %convert_element_type_52 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_24, torch.float32), kwargs = {})
#   %sigmoid_24 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_52,), kwargs = {})
#   %mul_1429 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_52, %sigmoid_24), kwargs = {})
#   %convert_element_type_53 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1429, torch.bfloat16), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 1170*x1 + 390*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), tmp14, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/2b/c2b53nyzvb5ksz2iz5rbvecuncp2ek64chn2b2ccoziermxanapp.py
# Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_26 => mean_26
#   output_tensor_12 => add_1842
#   pow_27 => pow_27
#   x_61 => cat_27
#   x_62 => convolution_29
#   x_66 => convolution_31
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_1688], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg66_1, %arg67_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg70_1, %arg71_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1842 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, %convolution_31), kwargs = {})
#   %pow_27 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1842, 2), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [1], True), kwargs = {})
triton_red_fused_add_cat_convolution_mean_pow_58 = async_compile.triton('triton_red_fused_add_cat_convolution_mean_pow_58', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_cat_convolution_mean_pow_58', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_cat_convolution_mean_pow_58(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp10 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 390*r0_2 + 49920*x1 + 390*r0_2*(triton_helpers.div_floor_integer((-1) + ks1,  8)) + 49920*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + 390*r0_2 + 49920*x1 + 390*r0_2*(triton_helpers.div_floor_integer((-1) + ks1,  8)) + 49920*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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
    tl.store(out_ptr0 + (x3), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/yg/cygyyewfmwr7kfqvxj4tabqnu5npnypp25fdxt5iushzl6ifuhra.py
# Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26, add_39, sqrt_26, hidden_states_65, hidden_states_66], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_39 => add_1857
#   hidden_states_65 => div_26
#   hidden_states_66 => convert_element_type_56
#   mean_26 => mean_26
#   output_tensor_12 => add_1842
#   pow_27 => pow_27
#   sqrt_26 => sqrt_26
#   x_61 => cat_27
#   x_62 => convolution_29
#   x_66 => convolution_31
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_1688], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg66_1, %arg67_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg70_1, %arg71_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1842 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, %convolution_31), kwargs = {})
#   %pow_27 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1842, 2), kwargs = {})
#   %mean_26 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_27, [1], True), kwargs = {})
#   %add_1857 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_26, 1e-08), kwargs = {})
#   %sqrt_26 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1857,), kwargs = {})
#   %div_26 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1842, %sqrt_26), kwargs = {})
#   %convert_element_type_56 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_26, torch.float32), kwargs = {})
triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_59 = async_compile.triton('triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_59', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*fp32', 'out_ptr0': '*fp32', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_59', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_59(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x2), tmp15, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/qf/cqfcptt6ariuiy47nk46v3jur6ncfhrgmvhjkqhjix2tx5yxvr6j.py
# Topologically Sorted Source Nodes: [x_67, x_68], Original ATen: [aten.cat, aten.convolution]
# Source node to ATen node mapping:
#   x_67 => cat_30
#   x_68 => convolution_32
# Graph fragment:
#   %cat_30 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_30, %convert_element_type_57], 2), kwargs = {})
#   %convolution_32 : [num_users=2] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_30, %arg72_1, %arg73_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
triton_poi_fused_cat_convolution_60 = async_compile.triton('triton_poi_fused_cat_convolution_60', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'ks2': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_60', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_convolution_60(in_ptr0, out_ptr0, ks0, ks1, ks2, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 390) % ks0)
    x0 = (xindex % 390)
    x2 = xindex // ks1
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 390*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks2,  8))), xmask & tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = ks0
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr0 + (x0 + 390*x2 + 390*((-2) + x1) + 390*x2*(triton_helpers.div_floor_integer((-1) + ks2,  8))), xmask & tmp11, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.sigmoid(tmp14)
    tmp16 = tmp14 * tmp15
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x3), tmp20, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/75/c75tteuawofbnrpkbfsu4z5y3fdndws2yaarajg7fr3zy2nkmwol.py
# Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, x_70, output_tensor_13], Original ATen: [aten.cat, aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_12 => add_1842
#   output_tensor_13 => add_1979
#   x_61 => cat_27
#   x_62 => convolution_29
#   x_66 => convolution_31
#   x_70 => convolution_33
# Graph fragment:
#   %cat_27 : [num_users=1] = call_function[target=torch.ops.aten.cat.default](args = ([%repeat_27, %add_1688], 2), kwargs = {})
#   %convolution_29 : [num_users=3] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_27, %arg66_1, %arg67_1, [2, 2, 2], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %convolution_31 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_29, %arg70_1, %arg71_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1842 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%convolution_29, %convolution_31), kwargs = {})
#   %convolution_33 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_31, %arg74_1, %arg75_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_1979 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1842, %convolution_33), kwargs = {})
triton_poi_fused_add_cat_convolution_61 = async_compile.triton('triton_poi_fused_add_cat_convolution_61', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'in_ptr4': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_cat_convolution_61', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_cat_convolution_61(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp9 = tmp7 + tmp8
    tmp10 = tmp6 + tmp9
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/y3/cy3ifp3bzsezv6kaj337z7frpyndaqtbrxlkhl76ds4bruxp4hgw.py
# Topologically Sorted Source Nodes: [pow_29, mean_28], Original ATen: [aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_28 => mean_28
#   pow_29 => pow_29
# Graph fragment:
#   %pow_29 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1979, 2), kwargs = {})
#   %mean_28 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_29, [1], True), kwargs = {})
triton_red_fused_mean_pow_62 = async_compile.triton('triton_red_fused_mean_pow_62', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_pow_62', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_mean_pow_62(in_ptr0, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp4 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 390*r0_2 + 49920*x1 + 390*r0_2*(triton_helpers.div_floor_integer((-1) + ks1,  8)) + 49920*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tmp0 * tmp0
        tmp2 = tmp1.to(tl.float32)
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(r0_mask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/5k/c5kski2ut5ilgh3qzhxjdm2pxusaufxijsg2zqqlpo4rfiojgdq4.py
# Topologically Sorted Source Nodes: [first_frame_pad_32], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_32 => repeat_32
# Graph fragment:
#   %repeat_32 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_163, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_63 = async_compile.triton('triton_poi_fused_repeat_63', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_63', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_63(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 399360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 390)
    x2 = xindex // 780
    x3 = (xindex % 780)
    tmp0 = tl.load(in_ptr0 + (x0 + 390*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks0,  8))), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 1170*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks0,  8))), tmp12, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/b4/cb4eg4dqnook7c4qfnf3gpjxbixp5vyz4dsx4d3cpmev2snnnpwb.py
# Topologically Sorted Source Nodes: [pow_29, mean_28, add_42, sqrt_28, hidden_states_70, hidden_states_71], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_42 => add_1994
#   hidden_states_70 => div_28
#   hidden_states_71 => convert_element_type_60, convert_element_type_61, mul_1645, sigmoid_28
#   mean_28 => mean_28
#   pow_29 => pow_29
#   sqrt_28 => sqrt_28
# Graph fragment:
#   %pow_29 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_1979, 2), kwargs = {})
#   %mean_28 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_29, [1], True), kwargs = {})
#   %add_1994 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_28, 1e-08), kwargs = {})
#   %sqrt_28 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_1994,), kwargs = {})
#   %div_28 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_1979, %sqrt_28), kwargs = {})
#   %convert_element_type_60 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_28, torch.float32), kwargs = {})
#   %sigmoid_28 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_60,), kwargs = {})
#   %mul_1645 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_60, %sigmoid_28), kwargs = {})
#   %convert_element_type_61 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1645, torch.bfloat16), kwargs = {})
triton_poi_fused_add_div_mean_pow_silu_sqrt_64 = async_compile.triton('triton_poi_fused_add_div_mean_pow_silu_sqrt_64', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_mean_pow_silu_sqrt_64', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_div_mean_pow_silu_sqrt_64(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 1170*x1 + 390*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), tmp12, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/us/cusunul4d7ksj4ybm6hwact64lwwsoxtgbd26ewabtotzxkpowto.py
# Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   mean_30 => mean_30
#   output_tensor_14 => add_2116
#   pow_31 => pow_31
#   x_74 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %arg78_1, %arg79_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_2116 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1979, %convolution_35), kwargs = {})
#   %pow_31 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2116, 2), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_31, [1], True), kwargs = {})
triton_red_fused_add_convolution_mean_pow_65 = async_compile.triton('triton_red_fused_add_convolution_mean_pow_65', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 4096, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr0': '*fp32', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_convolution_mean_pow_65', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 1, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_convolution_mean_pow_65(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ks0, ks1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % ks0)
    x1 = xindex // ks0
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 390*r0_2 + 49920*x1 + 390*r0_2*(triton_helpers.div_floor_integer((-1) + ks1,  8)) + 49920*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + 390*r0_2 + 49920*x1 + 390*r0_2*(triton_helpers.div_floor_integer((-1) + ks1,  8)) + 49920*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2 + 128*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/ze/czemnqcrdznj5x5cio2cixba7iyhmqqiob52w7uub7ve3o3uzcvz.py
# Topologically Sorted Source Nodes: [first_frame_pad_34], Original ATen: [aten.repeat]
# Source node to ATen node mapping:
#   first_frame_pad_34 => repeat_34
# Graph fragment:
#   %repeat_34 : [num_users=1] = call_function[target=torch.ops.aten.repeat.default](args = (%slice_173, [1, 1, 2, 1, 1]), kwargs = {})
triton_poi_fused_repeat_66 = async_compile.triton('triton_poi_fused_repeat_66', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_repeat_66', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_repeat_66(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 399360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 390)
    x2 = xindex // 780
    x3 = (xindex % 780)
    tmp0 = tl.load(in_ptr0 + (x0 + 390*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks0,  8))), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + 390*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks0,  8))), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x3 + 1170*x2 + 390*x2*(triton_helpers.div_floor_integer((-1) + ks0,  8))), tmp16, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/pr/cpricjgl5lwtu2e3no2f25e5xuvizwlh5yu7fpqf6x2tx4tafu5o.py
# Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30, add_45, sqrt_30, hidden_states_75, hidden_states_76], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
# Source node to ATen node mapping:
#   add_45 => add_2131
#   hidden_states_75 => div_30
#   hidden_states_76 => convert_element_type_64, convert_element_type_65, mul_1753, sigmoid_30
#   mean_30 => mean_30
#   output_tensor_14 => add_2116
#   pow_31 => pow_31
#   sqrt_30 => sqrt_30
#   x_74 => convolution_35
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %arg78_1, %arg79_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_2116 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1979, %convolution_35), kwargs = {})
#   %pow_31 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%add_2116, 2), kwargs = {})
#   %mean_30 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_31, [1], True), kwargs = {})
#   %add_2131 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mean_30, 1e-08), kwargs = {})
#   %sqrt_30 : [num_users=1] = call_function[target=torch.ops.aten.sqrt.default](args = (%add_2131,), kwargs = {})
#   %div_30 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%add_2116, %sqrt_30), kwargs = {})
#   %convert_element_type_64 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%div_30, torch.float32), kwargs = {})
#   %sigmoid_30 : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_64,), kwargs = {})
#   %mul_1753 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_64, %sigmoid_30), kwargs = {})
#   %convert_element_type_65 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_1753, torch.bfloat16), kwargs = {})
triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67 = async_compile.triton('triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*fp32', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    tmp0 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
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
    tl.store(out_ptr0 + (x0 + 1170*x1 + 390*x1*(triton_helpers.div_floor_integer((-1) + ks1,  8))), tmp16, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/aj/cajxrw7a77btyyw4bx5hmtll54d34qa246hwgaq3glgz2qi36j7h.py
# Topologically Sorted Source Nodes: [x_74, output_tensor_14, x_78, output_tensor_15], Original ATen: [aten.convolution, aten.add]
# Source node to ATen node mapping:
#   output_tensor_14 => add_2116
#   output_tensor_15 => add_2253
#   x_74 => convolution_35
#   x_78 => convolution_37
# Graph fragment:
#   %convolution_35 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_33, %arg78_1, %arg79_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_2116 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_1979, %convolution_35), kwargs = {})
#   %convolution_37 : [num_users=1] = call_function[target=torch.ops.aten.convolution.default](args = (%cat_35, %arg82_1, %arg83_1, [1, 1, 1], [0, 1, 1], [1, 1, 1], False, [0, 0, 0], 1), kwargs = {})
#   %add_2253 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_2116, %convolution_37), kwargs = {})
triton_poi_fused_add_convolution_68 = async_compile.triton('triton_poi_fused_add_convolution_68', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 524288}, 
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*bf16', 'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'in_ptr3': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_68', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_convolution_68(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = xindex // ks0
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/uc/cucuwxhh3nd2w5enwf2nlrhwwjqrjgzveuvgw2np6mdwihvl6ems.py
# Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.cat]
# Source node to ATen node mapping:
#   sample_2 => cat_43
# Graph fragment:
#   %cat_43 : [num_users=2] = call_function[target=torch.ops.aten.cat.default](args = ([%convolution_44, %repeat_43], 1), kwargs = {})
triton_poi_fused_cat_69 = async_compile.triton('triton_poi_fused_cat_69', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i32', 'ks1': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_69', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_cat_69(in_ptr0, in_ptr1, out_ptr0, ks0, ks1, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = xindex // ks0
    x0 = (xindex % ks0)
    x2 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 129, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0 + 390*(x1) + 390*(triton_helpers.div_floor_integer((-1) + ks1,  8))*(x1)), xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp6 = tl.load(in_ptr1 + (x1), xmask & tmp4, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 256, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tl.load(in_ptr0 + (49920 + x0 + 49920*(triton_helpers.div_floor_integer((-1) + ks1,  8))), xmask & tmp10, eviction_policy='evict_last', other=0.0).to(tl.float32)
    tmp14 = tl.load(in_ptr1 + (128)).to(tl.float32)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK])
    tmp16 = tl.where(tmp10, tmp15, 0.0)
    tmp17 = tmp13 + tmp16
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp10, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp9, tmp19)
    tl.store(out_ptr0 + (x2), tmp20, xmask)
''', device_str='cuda')


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/jv/cjv47kkszsu7ejf4eksjd7je3wvhfnh4v6aobmzscwvbxunal3lh.py
# Topologically Sorted Source Nodes: [clamp, exp_1, mul, exp], Original ATen: [aten.clamp, aten.exp, aten.mul]
# Source node to ATen node mapping:
#   clamp => clamp_max, clamp_min, convert_element_type_82, convert_element_type_83
#   exp => exp
#   exp_1 => exp_1
#   mul => mul_2243
# Graph fragment:
#   %convert_element_type_82 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%getitem_5, torch.float32), kwargs = {})
#   %clamp_min : [num_users=1] = call_function[target=torch.ops.aten.clamp_min.default](args = (%convert_element_type_82, -30.0), kwargs = {})
#   %clamp_max : [num_users=1] = call_function[target=torch.ops.aten.clamp_max.default](args = (%clamp_min, 20.0), kwargs = {})
#   %convert_element_type_83 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%clamp_max, torch.bfloat16), kwargs = {})
#   %exp_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%convert_element_type_83,), kwargs = {})
#   %mul_2243 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_83, 0.5), kwargs = {})
#   %exp : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%mul_2243,), kwargs = {})
triton_poi_fused_clamp_exp_mul_70 = async_compile.triton('triton_poi_fused_clamp_exp_mul_70', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'out_ptr0': '*bf16', 'out_ptr1': '*bf16', 'out_ptr2': '*bf16', 'ks0': 'i32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clamp_exp_mul_70', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clamp_exp_mul_70(in_ptr0, out_ptr0, out_ptr1, out_ptr2, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (49920 + x0 + 49920*(triton_helpers.div_floor_integer((-1) + ks0,  8))), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = -30.0
    tmp3 = triton_helpers.maximum(tmp1, tmp2)
    tmp4 = 20.0
    tmp5 = triton_helpers.minimum(tmp3, tmp4)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl_math.exp(tmp6)
    tmp8 = 0.5
    tmp9 = tmp6 * tmp8
    tmp10 = tl_math.exp(tmp9)
    tl.store(out_ptr0 + (x0), tmp6, xmask)
    tl.store(out_ptr1 + (x0), tmp7, xmask)
    tl.store(out_ptr2 + (x0), tmp10, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1 = args
    args.clear()
    s0 = arg0_1
    s1 = arg1_1
    s2 = arg2_1
    assert_size_stride(arg3_1, (1, 3, s0, 832, 480), (3*s1, s1, 399360, 480, 1))
    assert_size_stride(arg4_1, (128, 48, 3, 3, 3), (1296, 27, 9, 3, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg21_1, (128, ), (1, ))
    assert_size_stride(arg22_1, (128, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (256, 128, 3, 3, 3), (3456, 27, 9, 3, 1))
    assert_size_stride(arg25_1, (256, ), (1, ))
    assert_size_stride(arg26_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg27_1, (256, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (256, 128, 1, 1, 1), (128, 1, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg35_1, (256, ), (1, ))
    assert_size_stride(arg36_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg37_1, (256, ), (1, ))
    assert_size_stride(arg38_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (256, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (512, 256, 3, 3, 3), (6912, 27, 9, 3, 1))
    assert_size_stride(arg47_1, (512, ), (1, ))
    assert_size_stride(arg48_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg49_1, (512, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (512, 256, 1, 1, 1), (256, 1, 1, 1, 1))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg57_1, (512, ), (1, ))
    assert_size_stride(arg58_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg59_1, (512, ), (1, ))
    assert_size_stride(arg60_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg63_1, (512, ), (1, ))
    assert_size_stride(arg64_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg65_1, (512, ), (1, ))
    assert_size_stride(arg66_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg67_1, (512, ), (1, ))
    assert_size_stride(arg68_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg69_1, (512, ), (1, ))
    assert_size_stride(arg70_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg73_1, (512, ), (1, ))
    assert_size_stride(arg74_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg75_1, (512, ), (1, ))
    assert_size_stride(arg76_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg77_1, (512, ), (1, ))
    assert_size_stride(arg78_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg79_1, (512, ), (1, ))
    assert_size_stride(arg80_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg81_1, (512, ), (1, ))
    assert_size_stride(arg82_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg83_1, (512, ), (1, ))
    assert_size_stride(arg84_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg85_1, (512, ), (1, ))
    assert_size_stride(arg86_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg87_1, (512, ), (1, ))
    assert_size_stride(arg88_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg89_1, (512, ), (1, ))
    assert_size_stride(arg90_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg91_1, (512, ), (1, ))
    assert_size_stride(arg92_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (512, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (129, 512, 3, 3, 3), (13824, 27, 9, 3, 1))
    assert_size_stride(arg97_1, (129, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        ps0 = 2 + s0
        ps1 = 49920 + 24960*s0
        buf0 = empty_strided_cuda((1, 48, 2 + s0, 208, 120), (2396160 + 1198080*s0, 49920 + 24960*s0, 24960, 120, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_0_xnumel = 2396160 + 1198080*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_0.run(arg3_1, buf0, ps0, ps1, s1, triton_poi_fused_cat_convolution_0_xnumel, stream=stream0)
        del arg3_1
        # Topologically Sorted Source Nodes: [x_1, x_2], Original ATen: [aten.cat, aten.convolution]
        buf1 = extern_kernels.convolution(buf0, arg4_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf1, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg4_1
        del buf0
        buf2 = empty_strided_cuda((1, 1, s0, 208, 120), (24960*s0, 24960*s0, 24960, 120, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_1_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf1, arg5_1, buf2, s0, triton_red_fused_cat_convolution_mean_pow_1_xnumel, 128, stream=stream0)
        buf5 = empty_strided_cuda((1, 128, 2 + s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), torch.bfloat16)
        buf3 = reinterpret_tensor(buf5, (1, 128, 2, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_1], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf1, arg5_1, buf2, buf3, s0, 6389760, stream=stream0)
        ps2 = 24960*s0
        buf4 = reinterpret_tensor(buf5, (1, 128, s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_1, x_2, pow_1, mean, add, sqrt, hidden_states, hidden_states_1], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf1, arg5_1, buf2, buf4, ps2, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel, stream=stream0)
        del buf3
        del buf4
        # Topologically Sorted Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg6_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg6_1
        buf7 = buf2; del buf2  # reuse
        # Topologically Sorted Source Nodes: [x_4, pow_2, mean_1], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_1_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf6, arg7_1, buf7, s0, triton_red_fused_cat_convolution_mean_pow_1_xnumel, 128, stream=stream0)
        buf10 = buf5; del buf5  # reuse
        buf8 = reinterpret_tensor(buf10, (1, 128, 2, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_2], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf6, arg7_1, buf7, buf8, s0, 6389760, stream=stream0)
        buf9 = reinterpret_tensor(buf10, (1, 128, s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_4, pow_2, mean_1, add_1, sqrt_1, hidden_states_2, hidden_states_3], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf6, arg7_1, buf7, buf9, ps2, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel, stream=stream0)
        del arg7_1
        del buf6
        del buf8
        del buf9
        # Topologically Sorted Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg8_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg8_1
        buf12 = buf7; del buf7  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_cat_convolution_mean_pow_4_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_convolution_mean_pow_4.run(buf1, arg5_1, buf11, arg9_1, buf12, s0, triton_red_fused_add_cat_convolution_mean_pow_4_xnumel, 128, stream=stream0)
        buf13 = empty_strided_cuda((1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, pow_3, mean_2, add_3, sqrt_2, hidden_states_5, hidden_states_6], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5.run(buf1, arg5_1, buf11, arg9_1, buf12, buf13, ps2, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_5_xnumel, stream=stream0)
        buf14 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_6_xnumel = 6389760 + 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_6.run(buf13, buf14, ps0, ps1, s0, triton_poi_fused_cat_convolution_6_xnumel, stream=stream0)
        del buf13
        # Topologically Sorted Source Nodes: [x_7, x_8], Original ATen: [aten.cat, aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg10_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg10_1
        buf16 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [x_7, x_8, pow_4, mean_3], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_1_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf15, arg11_1, buf16, s0, triton_red_fused_cat_convolution_mean_pow_1_xnumel, 128, stream=stream0)
        buf19 = buf14; del buf14  # reuse
        buf17 = reinterpret_tensor(buf19, (1, 128, 2, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_4], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf15, arg11_1, buf16, buf17, s0, 6389760, stream=stream0)
        buf18 = reinterpret_tensor(buf19, (1, 128, s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_7, x_8, pow_4, mean_3, add_4, sqrt_3, hidden_states_7, hidden_states_8], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf15, arg11_1, buf16, buf18, ps2, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel, stream=stream0)
        del arg11_1
        del buf15
        del buf17
        del buf18
        # Topologically Sorted Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg12_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg12_1
        buf21 = buf1; del buf1  # reuse
        # Topologically Sorted Source Nodes: [x_1, x_2, x_6, output_tensor, x_10, output_tensor_1], Original ATen: [aten.cat, aten.convolution, aten.add]
        triton_poi_fused_add_cat_convolution_7_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_7.run(buf21, arg5_1, buf11, arg9_1, buf20, arg13_1, ps2, triton_poi_fused_add_cat_convolution_7_xnumel, stream=stream0)
        del arg13_1
        del arg5_1
        del arg9_1
        del buf11
        del buf20
        buf22 = buf16; del buf16  # reuse
        # Topologically Sorted Source Nodes: [pow_5, mean_4], Original ATen: [aten.pow, aten.mean]
        triton_red_fused_mean_pow_8_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_8.run(buf21, buf22, s0, triton_red_fused_mean_pow_8_xnumel, 128, stream=stream0)
        buf25 = buf19; del buf19  # reuse
        buf23 = reinterpret_tensor(buf25, (1, 128, 2, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_5], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_9.run(buf21, buf22, buf23, s0, 6389760, stream=stream0)
        buf24 = reinterpret_tensor(buf25, (1, 128, s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [pow_5, mean_4, add_6, sqrt_4, hidden_states_10, hidden_states_11], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_div_mean_pow_silu_sqrt_10_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_10.run(buf21, buf22, buf24, ps2, s0, triton_poi_fused_add_div_mean_pow_silu_sqrt_10_xnumel, stream=stream0)
        del buf23
        del buf24
        # Topologically Sorted Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg14_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg14_1
        buf27 = buf22; del buf22  # reuse
        # Topologically Sorted Source Nodes: [x_12, pow_6, mean_5], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_1_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf26, arg15_1, buf27, s0, triton_red_fused_cat_convolution_mean_pow_1_xnumel, 128, stream=stream0)
        buf30 = buf25; del buf25  # reuse
        buf28 = reinterpret_tensor(buf30, (1, 128, 2, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_6], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf26, arg15_1, buf27, buf28, s0, 6389760, stream=stream0)
        buf29 = reinterpret_tensor(buf30, (1, 128, s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_12, pow_6, mean_5, add_7, sqrt_5, hidden_states_12, hidden_states_13], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf26, arg15_1, buf27, buf29, ps2, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel, stream=stream0)
        del arg15_1
        del buf26
        del buf28
        del buf29
        # Topologically Sorted Source Nodes: [x_14], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg16_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg16_1
        buf32 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_11_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_11.run(buf21, buf31, arg17_1, buf32, s0, triton_red_fused_add_convolution_mean_pow_11_xnumel, 128, stream=stream0)
        buf35 = buf30; del buf30  # reuse
        buf33 = reinterpret_tensor(buf35, (1, 128, 2, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_7], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_12.run(buf21, buf31, arg17_1, buf32, buf33, s0, 6389760, stream=stream0)
        buf34 = reinterpret_tensor(buf35, (1, 128, s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_14, output_tensor_2, pow_7, mean_6, add_9, sqrt_6, hidden_states_15, hidden_states_16], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13.run(buf21, buf31, arg17_1, buf32, buf34, ps2, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_13_xnumel, stream=stream0)
        del buf33
        del buf34
        # Topologically Sorted Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg18_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg18_1
        buf37 = buf32; del buf32  # reuse
        # Topologically Sorted Source Nodes: [x_16, pow_8, mean_7], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_1_xnumel = 24960*s0
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_1.run(buf36, arg19_1, buf37, s0, triton_red_fused_cat_convolution_mean_pow_1_xnumel, 128, stream=stream0)
        buf40 = buf35; del buf35  # reuse
        buf38 = reinterpret_tensor(buf40, (1, 128, 2, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_8], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_2.run(buf36, arg19_1, buf37, buf38, s0, 6389760, stream=stream0)
        buf39 = reinterpret_tensor(buf40, (1, 128, s0, 208, 120), (6389760 + 3194880*s0, 49920 + 24960*s0, 24960, 120, 1), 49920)  # alias
        # Topologically Sorted Source Nodes: [x_16, pow_8, mean_7, add_10, sqrt_7, hidden_states_17, hidden_states_18], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3.run(buf36, arg19_1, buf37, buf39, ps2, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_3_xnumel, stream=stream0)
        del arg19_1
        del buf36
        del buf37
        del buf38
        del buf39
        # Topologically Sorted Source Nodes: [x_18], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg20_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (1, 128, s0, 208, 120), (3194880*s0, 24960*s0, 24960, 120, 1))
        del arg20_1
        buf42 = buf21; del buf21  # reuse
        # Topologically Sorted Source Nodes: [x_14, output_tensor_2, x_18, output_tensor_3], Original ATen: [aten.convolution, aten.add]
        triton_poi_fused_add_convolution_14_xnumel = 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_14.run(buf42, buf31, arg17_1, buf41, arg21_1, ps2, triton_poi_fused_add_convolution_14_xnumel, stream=stream0)
        del arg17_1
        del arg21_1
        del buf31
        del buf41
        buf43 = buf40; del buf40  # reuse
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_15_xnumel = 6389760 + 3194880*s0
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_15.run(buf42, buf43, ps0, ps1, s0, triton_poi_fused_cat_convolution_15_xnumel, stream=stream0)
        del buf42
        # Topologically Sorted Source Nodes: [x_19, x_20], Original ATen: [aten.cat, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg22_1, stride=(2, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (1, 128, 1 + (((-1) + s0) // 2), 104, 60), (798720 + 798720*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg22_1
        del buf43
        buf45 = empty_strided_cuda((1, 1 + (((-1) + s0) // 2), 104, 60, 1), (6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1, 6240 + 6240*(((-1) + s0) // 2)), torch.float32)
        buf46 = empty_strided_cuda((1, 1 + (((-1) + s0) // 2), 104, 60, 1), (6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1, 6240 + 6240*(((-1) + s0) // 2)), torch.float32)
        buf50 = empty_strided_cuda((1, 1, 1 + (((-1) + s0) // 2), 104, 60), (6240 + 6240*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_19, x_20, x_26, pow_9, mean_8], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16.run(buf44, arg23_1, buf45, buf46, buf50, s0, triton_red_fused_cat_convolution_mean_native_layer_norm_pow_16_xnumel, 128, stream=stream0)
        buf48 = empty_strided_cuda((1, 128, 1 + (((-1) + s0) // 2), 104, 60), (798720 + 798720*(((-1) + s0) // 2), 1, 798720, 7680, 128), torch.bfloat16)
        buf53 = empty_strided_cuda((1, 128, 3 + (((-1) + s0) // 2), 104, 60), (2396160 + 798720*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), torch.bfloat16)
        buf52 = reinterpret_tensor(buf53, (1, 128, 1 + (((-1) + s0) // 2), 104, 60), (2396160 + 798720*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_19, x_20, input_tensor, pow_9, mean_8, add_12, sqrt_8, hidden_states_20, hidden_states_21], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_17_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_17.run(buf44, arg23_1, buf45, buf46, arg28_1, arg29_1, buf50, buf48, buf52, s0, 128, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_17_xnumel, stream=stream0)
        del arg28_1
        del arg29_1
        del buf45
        del buf46
        # Topologically Sorted Source Nodes: [input_tensor], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg30_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 1, 1597440, 15360, 256))
        del arg30_1
        del buf48
        buf51 = reinterpret_tensor(buf53, (1, 128, 2, 104, 60), (2396160 + 798720*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_10], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_18.run(buf44, arg23_1, buf50, buf51, s0, 1597440, stream=stream0)
        del arg23_1
        del buf44
        del buf51
        del buf52
        # Topologically Sorted Source Nodes: [x_22], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, arg24_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg24_1
        del buf53
        buf55 = buf50; del buf50  # reuse
        # Topologically Sorted Source Nodes: [x_22, pow_10, mean_9], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_19_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_19.run(buf54, arg25_1, buf55, s0, triton_red_fused_convolution_mean_pow_19_xnumel, 256, stream=stream0)
        buf58 = empty_strided_cuda((1, 256, 3 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), torch.bfloat16)
        buf56 = reinterpret_tensor(buf58, (1, 256, 2, 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_11], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_20.run(buf54, arg25_1, buf55, buf56, s0, 3194880, stream=stream0)
        ps3 = 6240 + 6240*(((-1) + s0) // 2)
        buf57 = reinterpret_tensor(buf58, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_22, pow_10, mean_9, add_13, sqrt_9, hidden_states_22, hidden_states_23], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel = 1597440 + 1597440*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21.run(buf54, arg25_1, buf55, buf57, ps3, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel, stream=stream0)
        del arg25_1
        del buf54
        del buf56
        del buf57
        # Topologically Sorted Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, arg26_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg26_1
        buf60 = buf55; del buf55  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_22_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_22.run(buf49, arg31_1, buf59, arg27_1, buf60, s0, triton_red_fused_add_convolution_mean_pow_22_xnumel, 256, stream=stream0)
        buf61 = empty_strided_cuda((1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, pow_11, mean_10, add_15, sqrt_10, hidden_states_25, hidden_states_26], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_23_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_23.run(buf49, arg31_1, buf59, arg27_1, buf60, buf61, s0, 256, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_23_xnumel, stream=stream0)
        ps4 = 3 + (((-1) + s0) // 2)
        ps5 = 18720 + 6240*(((-1) + s0) // 2)
        buf62 = buf58; del buf58  # reuse
        # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_24_xnumel = 4792320 + 1597440*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_24.run(buf61, buf62, ps4, ps5, s0, triton_poi_fused_cat_convolution_24_xnumel, stream=stream0)
        del buf61
        # Topologically Sorted Source Nodes: [x_28, x_29], Original ATen: [aten.cat, aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg32_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg32_1
        buf64 = buf60; del buf60  # reuse
        # Topologically Sorted Source Nodes: [x_28, x_29, pow_12, mean_11], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_19_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_19.run(buf63, arg33_1, buf64, s0, triton_red_fused_convolution_mean_pow_19_xnumel, 256, stream=stream0)
        buf67 = buf62; del buf62  # reuse
        buf65 = reinterpret_tensor(buf67, (1, 256, 2, 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_13], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_20.run(buf63, arg33_1, buf64, buf65, s0, 3194880, stream=stream0)
        buf66 = reinterpret_tensor(buf67, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_28, x_29, pow_12, mean_11, add_16, sqrt_11, hidden_states_27, hidden_states_28], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel = 1597440 + 1597440*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21.run(buf63, arg33_1, buf64, buf66, ps3, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel, stream=stream0)
        del arg33_1
        del buf63
        del buf65
        del buf66
        # Topologically Sorted Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg34_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg34_1
        buf69 = buf59; del buf59  # reuse
        # Topologically Sorted Source Nodes: [input_tensor, x_24, output_tensor_4, x_31, output_tensor_5], Original ATen: [aten.convolution, aten.add]
        triton_poi_fused_add_convolution_25_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_25.run(buf69, buf49, arg31_1, arg27_1, buf68, arg35_1, s0, 256, triton_poi_fused_add_convolution_25_xnumel, stream=stream0)
        del arg27_1
        del arg31_1
        del arg35_1
        del buf49
        del buf68
        buf70 = buf64; del buf64  # reuse
        # Topologically Sorted Source Nodes: [pow_13, mean_12], Original ATen: [aten.pow, aten.mean]
        triton_red_fused_mean_pow_26_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_26.run(buf69, buf70, s0, triton_red_fused_mean_pow_26_xnumel, 256, stream=stream0)
        buf73 = reinterpret_tensor(buf67, (1, 256, 3 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 1, 1597440, 15360, 256), 0); del buf67  # reuse
        buf71 = reinterpret_tensor(buf73, (1, 256, 2, 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 1, 1597440, 15360, 256), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_14], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_27.run(buf69, buf70, buf71, s0, 512, 6240, stream=stream0)
        buf72 = reinterpret_tensor(buf73, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 1, 1597440, 15360, 256), 3194880)  # alias
        # Topologically Sorted Source Nodes: [pow_13, mean_12, add_18, sqrt_12, hidden_states_30, hidden_states_31], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_div_mean_pow_silu_sqrt_28_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_28.run(buf69, buf70, buf72, s0, 256, triton_poi_fused_add_div_mean_pow_silu_sqrt_28_xnumel, stream=stream0)
        buf74 = empty_strided_cuda((1, 256, 3 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29_xnumel = 18720 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf73, buf74, s0, 256, triton_poi_fused_convolution_29_xnumel, stream=stream0)
        del buf71
        del buf72
        # Topologically Sorted Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg36_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg36_1
        buf76 = buf70; del buf70  # reuse
        # Topologically Sorted Source Nodes: [x_33, pow_14, mean_13], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_19_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_19.run(buf75, arg37_1, buf76, s0, triton_red_fused_convolution_mean_pow_19_xnumel, 256, stream=stream0)
        buf79 = buf74; del buf74  # reuse
        buf77 = reinterpret_tensor(buf79, (1, 256, 2, 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_15], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_20.run(buf75, arg37_1, buf76, buf77, s0, 3194880, stream=stream0)
        buf78 = reinterpret_tensor(buf79, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_33, pow_14, mean_13, add_19, sqrt_13, hidden_states_32, hidden_states_33], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel = 1597440 + 1597440*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21.run(buf75, arg37_1, buf76, buf78, ps3, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel, stream=stream0)
        del arg37_1
        del buf75
        del buf77
        del buf78
        # Topologically Sorted Source Nodes: [x_35], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg38_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg38_1
        buf81 = buf76; del buf76  # reuse
        # Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_30_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_30.run(buf69, buf80, arg39_1, buf81, s0, triton_red_fused_add_convolution_mean_pow_30_xnumel, 256, stream=stream0)
        buf84 = reinterpret_tensor(buf79, (1, 256, 3 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 1, 1597440, 15360, 256), 0); del buf79  # reuse
        buf82 = reinterpret_tensor(buf84, (1, 256, 2, 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 1, 1597440, 15360, 256), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_16], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_31.run(buf69, buf80, arg39_1, buf81, buf82, s0, 512, 6240, stream=stream0)
        buf83 = reinterpret_tensor(buf84, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 1, 1597440, 15360, 256), 3194880)  # alias
        # Topologically Sorted Source Nodes: [x_35, output_tensor_6, pow_15, mean_14, add_21, sqrt_14, hidden_states_35, hidden_states_36], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_32_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_32.run(buf69, buf80, arg39_1, buf81, buf83, s0, 256, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_32_xnumel, stream=stream0)
        buf85 = reinterpret_tensor(buf73, (1, 256, 3 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 0); del buf73  # reuse
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_29_xnumel = 18720 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_29.run(buf84, buf85, s0, 256, triton_poi_fused_convolution_29_xnumel, stream=stream0)
        del buf82
        del buf83
        del buf84
        # Topologically Sorted Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg40_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg40_1
        buf87 = buf81; del buf81  # reuse
        # Topologically Sorted Source Nodes: [x_37, pow_16, mean_15], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_19_xnumel = 6240 + 6240*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_19.run(buf86, arg41_1, buf87, s0, triton_red_fused_convolution_mean_pow_19_xnumel, 256, stream=stream0)
        buf90 = buf85; del buf85  # reuse
        buf88 = reinterpret_tensor(buf90, (1, 256, 2, 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_17], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_20.run(buf86, arg41_1, buf87, buf88, s0, 3194880, stream=stream0)
        buf89 = reinterpret_tensor(buf90, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (4792320 + 1597440*(((-1) + s0) // 2), 18720 + 6240*(((-1) + s0) // 2), 6240, 60, 1), 12480)  # alias
        # Topologically Sorted Source Nodes: [x_37, pow_16, mean_15, add_22, sqrt_15, hidden_states_37, hidden_states_38], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel = 1597440 + 1597440*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21.run(buf86, arg41_1, buf87, buf89, ps3, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_21_xnumel, stream=stream0)
        del arg41_1
        del buf86
        del buf87
        del buf88
        del buf89
        # Topologically Sorted Source Nodes: [x_39], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg42_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (1, 256, 1 + (((-1) + s0) // 2), 104, 60), (1597440 + 1597440*(((-1) + s0) // 2), 6240 + 6240*(((-1) + s0) // 2), 6240, 60, 1))
        del arg42_1
        buf92 = buf69; del buf69  # reuse
        # Topologically Sorted Source Nodes: [x_35, output_tensor_6, x_39, output_tensor_7], Original ATen: [aten.convolution, aten.add]
        triton_poi_fused_add_convolution_33_xnumel = 1597440 + 1597440*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_33.run(buf92, buf80, arg39_1, buf91, arg43_1, ps3, triton_poi_fused_add_convolution_33_xnumel, stream=stream0)
        del arg39_1
        del arg43_1
        del buf80
        del buf91
        buf93 = buf90; del buf90  # reuse
        # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_34_xnumel = 4792320 + 1597440*(((-1) + s0) // 2)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_34.run(buf92, buf93, ps4, ps5, s0, triton_poi_fused_cat_convolution_34_xnumel, stream=stream0)
        del buf92
        # Topologically Sorted Source Nodes: [x_40, x_41], Original ATen: [aten.cat, aten.convolution]
        buf94 = extern_kernels.convolution(buf93, arg44_1, stride=(2, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (1, 256, 1 + (((-1) + s0) // 4), 52, 30), (399360 + 399360*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg44_1
        del buf93
        buf95 = empty_strided_cuda((1, 1 + (((-1) + s0) // 4), 52, 30, 1), (1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1, 1560 + 1560*(((-1) + s0) // 4)), torch.float32)
        buf96 = empty_strided_cuda((1, 1 + (((-1) + s0) // 4), 52, 30, 1), (1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1, 1560 + 1560*(((-1) + s0) // 4)), torch.float32)
        buf100 = empty_strided_cuda((1, 1, 1 + (((-1) + s0) // 4), 52, 30), (1560 + 1560*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_40, x_41, x_47, pow_17, mean_16], Original ATen: [aten.cat, aten.convolution, aten.native_layer_norm, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_native_layer_norm_pow_35_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_native_layer_norm_pow_35.run(buf94, arg45_1, buf95, buf96, buf100, s0, triton_red_fused_cat_convolution_mean_native_layer_norm_pow_35_xnumel, 256, stream=stream0)
        buf98 = empty_strided_cuda((1, 256, 1 + (((-1) + s0) // 4), 52, 30), (399360 + 399360*(((-1) + s0) // 4), 1, 399360, 7680, 256), torch.bfloat16)
        buf103 = empty_strided_cuda((1, 256, 3 + (((-1) + s0) // 4), 52, 30), (1198080 + 399360*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), torch.bfloat16)
        buf102 = reinterpret_tensor(buf103, (1, 256, 1 + (((-1) + s0) // 4), 52, 30), (1198080 + 399360*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_40, x_41, input_tensor_1, pow_17, mean_16, add_24, sqrt_16, hidden_states_40, hidden_states_41], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_36_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_36.run(buf94, arg45_1, buf95, buf96, arg50_1, arg51_1, buf100, buf98, buf102, s0, 256, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_36_xnumel, stream=stream0)
        del arg50_1
        del arg51_1
        del buf95
        del buf96
        # Topologically Sorted Source Nodes: [input_tensor_1], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg52_1, stride=(1, 1, 1), padding=(0, 0, 0), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1, 798720, 15360, 512))
        del arg52_1
        del buf98
        buf101 = reinterpret_tensor(buf103, (1, 256, 2, 52, 30), (1198080 + 399360*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_19], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_37.run(buf94, arg45_1, buf100, buf101, s0, 798720, stream=stream0)
        del arg45_1
        del buf94
        del buf101
        del buf102
        # Topologically Sorted Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg46_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg46_1
        del buf103
        buf105 = buf100; del buf100  # reuse
        # Topologically Sorted Source Nodes: [x_43, pow_18, mean_17], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_38_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_38.run(buf104, arg47_1, buf105, s0, triton_red_fused_convolution_mean_pow_38_xnumel, 512, stream=stream0)
        buf108 = empty_strided_cuda((1, 512, 3 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), torch.bfloat16)
        buf106 = reinterpret_tensor(buf108, (1, 512, 2, 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_20], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_39.run(buf104, arg47_1, buf105, buf106, s0, 1597440, stream=stream0)
        ps6 = 1560 + 1560*(((-1) + s0) // 4)
        buf107 = reinterpret_tensor(buf108, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_43, pow_18, mean_17, add_25, sqrt_17, hidden_states_42, hidden_states_43], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel = 798720 + 798720*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40.run(buf104, arg47_1, buf105, buf107, ps6, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel, stream=stream0)
        del arg47_1
        del buf104
        del buf106
        del buf107
        # Topologically Sorted Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf109 = extern_kernels.convolution(buf108, arg48_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg48_1
        buf110 = buf105; del buf105  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_41_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_41.run(buf99, arg53_1, buf109, arg49_1, buf110, s0, triton_red_fused_add_convolution_mean_pow_41_xnumel, 512, stream=stream0)
        buf111 = empty_strided_cuda((1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1), torch.float32)
        # Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, pow_19, mean_18, add_27, sqrt_18, hidden_states_45, hidden_states_46], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_42_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_42.run(buf99, arg53_1, buf109, arg49_1, buf110, buf111, s0, 512, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_42_xnumel, stream=stream0)
        ps7 = 3 + (((-1) + s0) // 4)
        ps8 = 4680 + 1560*(((-1) + s0) // 4)
        buf112 = buf108; del buf108  # reuse
        # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_43_xnumel = 2396160 + 798720*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_43.run(buf111, buf112, ps7, ps8, s0, triton_poi_fused_cat_convolution_43_xnumel, stream=stream0)
        del buf111
        # Topologically Sorted Source Nodes: [x_49, x_50], Original ATen: [aten.cat, aten.convolution]
        buf113 = extern_kernels.convolution(buf112, arg54_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg54_1
        buf114 = buf110; del buf110  # reuse
        # Topologically Sorted Source Nodes: [x_49, x_50, pow_20, mean_19], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_38_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_38.run(buf113, arg55_1, buf114, s0, triton_red_fused_convolution_mean_pow_38_xnumel, 512, stream=stream0)
        buf117 = buf112; del buf112  # reuse
        buf115 = reinterpret_tensor(buf117, (1, 512, 2, 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_22], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_39.run(buf113, arg55_1, buf114, buf115, s0, 1597440, stream=stream0)
        buf116 = reinterpret_tensor(buf117, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_49, x_50, pow_20, mean_19, add_28, sqrt_19, hidden_states_47, hidden_states_48], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel = 798720 + 798720*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40.run(buf113, arg55_1, buf114, buf116, ps6, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel, stream=stream0)
        del arg55_1
        del buf113
        del buf115
        del buf116
        # Topologically Sorted Source Nodes: [x_52], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, arg56_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg56_1
        buf119 = buf109; del buf109  # reuse
        # Topologically Sorted Source Nodes: [input_tensor_1, x_45, output_tensor_8, x_52, output_tensor_9], Original ATen: [aten.convolution, aten.add]
        triton_poi_fused_add_convolution_44_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_44.run(buf119, buf99, arg53_1, arg49_1, buf118, arg57_1, s0, 512, triton_poi_fused_add_convolution_44_xnumel, stream=stream0)
        del arg49_1
        del arg53_1
        del arg57_1
        del buf118
        del buf99
        buf120 = buf114; del buf114  # reuse
        # Topologically Sorted Source Nodes: [pow_21, mean_20], Original ATen: [aten.pow, aten.mean]
        triton_red_fused_mean_pow_45_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_45.run(buf119, buf120, s0, triton_red_fused_mean_pow_45_xnumel, 512, stream=stream0)
        buf123 = reinterpret_tensor(buf117, (1, 512, 3 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 1, 798720, 15360, 512), 0); del buf117  # reuse
        buf121 = reinterpret_tensor(buf123, (1, 512, 2, 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 1, 798720, 15360, 512), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_23], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_46.run(buf119, buf120, buf121, s0, 1024, 1560, stream=stream0)
        buf122 = reinterpret_tensor(buf123, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 1, 798720, 15360, 512), 1597440)  # alias
        # Topologically Sorted Source Nodes: [pow_21, mean_20, add_30, sqrt_20, hidden_states_50, hidden_states_51], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_div_mean_pow_silu_sqrt_47_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_47.run(buf119, buf120, buf122, s0, 512, triton_poi_fused_add_div_mean_pow_silu_sqrt_47_xnumel, stream=stream0)
        buf124 = empty_strided_cuda((1, 512, 3 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48_xnumel = 4680 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_48.run(buf123, buf124, s0, 512, triton_poi_fused_convolution_48_xnumel, stream=stream0)
        del buf121
        del buf122
        # Topologically Sorted Source Nodes: [x_54], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, arg58_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg58_1
        buf126 = buf120; del buf120  # reuse
        # Topologically Sorted Source Nodes: [x_54, pow_22, mean_21], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_38_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_38.run(buf125, arg59_1, buf126, s0, triton_red_fused_convolution_mean_pow_38_xnumel, 512, stream=stream0)
        buf129 = buf124; del buf124  # reuse
        buf127 = reinterpret_tensor(buf129, (1, 512, 2, 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_24], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_39.run(buf125, arg59_1, buf126, buf127, s0, 1597440, stream=stream0)
        buf128 = reinterpret_tensor(buf129, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_54, pow_22, mean_21, add_31, sqrt_21, hidden_states_52, hidden_states_53], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel = 798720 + 798720*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40.run(buf125, arg59_1, buf126, buf128, ps6, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel, stream=stream0)
        del arg59_1
        del buf125
        del buf127
        del buf128
        # Topologically Sorted Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf130 = extern_kernels.convolution(buf129, arg60_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg60_1
        buf131 = buf126; del buf126  # reuse
        # Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_49_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_49.run(buf119, buf130, arg61_1, buf131, s0, triton_red_fused_add_convolution_mean_pow_49_xnumel, 512, stream=stream0)
        buf134 = reinterpret_tensor(buf129, (1, 512, 3 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 1, 798720, 15360, 512), 0); del buf129  # reuse
        buf132 = reinterpret_tensor(buf134, (1, 512, 2, 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 1, 798720, 15360, 512), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_25], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_50.run(buf119, buf130, arg61_1, buf131, buf132, s0, 1024, 1560, stream=stream0)
        buf133 = reinterpret_tensor(buf134, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 1, 798720, 15360, 512), 1597440)  # alias
        # Topologically Sorted Source Nodes: [x_56, output_tensor_10, pow_23, mean_22, add_33, sqrt_22, hidden_states_55, hidden_states_56], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_51_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_51.run(buf119, buf130, arg61_1, buf131, buf133, s0, 512, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_51_xnumel, stream=stream0)
        buf135 = reinterpret_tensor(buf123, (1, 512, 3 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 0); del buf123  # reuse
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_48_xnumel = 4680 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_convolution_48.run(buf134, buf135, s0, 512, triton_poi_fused_convolution_48_xnumel, stream=stream0)
        del buf132
        del buf133
        del buf134
        # Topologically Sorted Source Nodes: [x_58], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, arg62_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf136, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg62_1
        buf137 = buf131; del buf131  # reuse
        # Topologically Sorted Source Nodes: [x_58, pow_24, mean_23], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_convolution_mean_pow_38_xnumel = 1560 + 1560*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_red_fused_convolution_mean_pow_38.run(buf136, arg63_1, buf137, s0, triton_red_fused_convolution_mean_pow_38_xnumel, 512, stream=stream0)
        buf140 = buf135; del buf135  # reuse
        buf138 = reinterpret_tensor(buf140, (1, 512, 2, 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_26], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_39.run(buf136, arg63_1, buf137, buf138, s0, 1597440, stream=stream0)
        buf139 = reinterpret_tensor(buf140, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (2396160 + 798720*(((-1) + s0) // 4), 4680 + 1560*(((-1) + s0) // 4), 1560, 30, 1), 3120)  # alias
        # Topologically Sorted Source Nodes: [x_58, pow_24, mean_23, add_34, sqrt_23, hidden_states_57, hidden_states_58], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel = 798720 + 798720*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40.run(buf136, arg63_1, buf137, buf139, ps6, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_40_xnumel, stream=stream0)
        del arg63_1
        del buf136
        del buf137
        del buf138
        del buf139
        # Topologically Sorted Source Nodes: [x_60], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg64_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (1, 512, 1 + (((-1) + s0) // 4), 52, 30), (798720 + 798720*(((-1) + s0) // 4), 1560 + 1560*(((-1) + s0) // 4), 1560, 30, 1))
        del arg64_1
        buf142 = buf119; del buf119  # reuse
        # Topologically Sorted Source Nodes: [x_56, output_tensor_10, x_60, output_tensor_11], Original ATen: [aten.convolution, aten.add]
        triton_poi_fused_add_convolution_52_xnumel = 798720 + 798720*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_52.run(buf142, buf130, arg61_1, buf141, arg65_1, ps6, triton_poi_fused_add_convolution_52_xnumel, stream=stream0)
        del arg61_1
        del arg65_1
        del buf130
        del buf141
        buf143 = buf140; del buf140  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_53_xnumel = 2396160 + 798720*(((-1) + s0) // 4)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_53.run(buf142, buf143, ps7, ps8, s0, triton_poi_fused_cat_convolution_53_xnumel, stream=stream0)
        del buf142
        # Topologically Sorted Source Nodes: [x_61, x_62], Original ATen: [aten.cat, aten.convolution]
        buf144 = extern_kernels.convolution(buf143, arg66_1, stride=(2, 2, 2), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg66_1
        del buf143
        ps9 = 390 + 390*(((-1) + s0) // 8)
        buf145 = empty_strided_cuda((1, 1, 1 + (((-1) + s0) // 8), 26, 15, 4), (1560 + 1560*(((-1) + s0) // 8), 1560 + 1560*(((-1) + s0) // 8), 390, 15, 1, 390 + 390*(((-1) + s0) // 8)), torch.float32)
        # Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf144, arg67_1, buf145, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf146 = empty_strided_cuda((1, 1, 1 + (((-1) + s0) // 8), 26, 15), (390 + 390*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf145, buf146, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf149 = empty_strided_cuda((1, 512, 3 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), torch.bfloat16)
        buf147 = reinterpret_tensor(buf149, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_28], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf144, arg67_1, buf146, buf147, s0, 399360, stream=stream0)
        buf148 = reinterpret_tensor(buf149, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_61, x_62, pow_25, mean_24, add_36, sqrt_24, hidden_states_60, hidden_states_61], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf144, arg67_1, buf146, buf148, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del buf147
        del buf148
        # Topologically Sorted Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg68_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg68_1
        buf151 = buf145; del buf145  # reuse
        # Topologically Sorted Source Nodes: [x_64, pow_26, mean_25], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf150, arg69_1, buf151, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf152 = buf146; del buf146  # reuse
        # Topologically Sorted Source Nodes: [x_64, pow_26, mean_25], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf151, buf152, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf155 = buf149; del buf149  # reuse
        buf153 = reinterpret_tensor(buf155, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_29], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf150, arg69_1, buf152, buf153, s0, 399360, stream=stream0)
        buf154 = reinterpret_tensor(buf155, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_64, pow_26, mean_25, add_37, sqrt_25, hidden_states_62, hidden_states_63], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf150, arg69_1, buf152, buf154, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del arg69_1
        del buf150
        del buf153
        del buf154
        # Topologically Sorted Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, arg70_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg70_1
        buf157 = buf151; del buf151  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_cat_convolution_mean_pow_58_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_cat_convolution_mean_pow_58.run(buf144, arg67_1, buf156, arg71_1, buf157, ps9, s0, triton_red_fused_add_cat_convolution_mean_pow_58_xnumel, 128, stream=stream0)
        buf158 = buf152; del buf152  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf157, buf158, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf159 = empty_strided_cuda((1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1), torch.float32)
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, pow_27, mean_26, add_39, sqrt_26, hidden_states_65, hidden_states_66], Original ATen: [aten.cat, aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_59_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_59.run(buf144, arg67_1, buf156, arg71_1, buf158, buf159, ps9, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_59_xnumel, stream=stream0)
        ps10 = 3 + (((-1) + s0) // 8)
        ps11 = 1170 + 390*(((-1) + s0) // 8)
        buf160 = buf155; del buf155  # reuse
        # Topologically Sorted Source Nodes: [x_67, x_68], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_60_xnumel = 599040 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_convolution_60.run(buf159, buf160, ps10, ps11, s0, triton_poi_fused_cat_convolution_60_xnumel, stream=stream0)
        del buf159
        # Topologically Sorted Source Nodes: [x_67, x_68], Original ATen: [aten.cat, aten.convolution]
        buf161 = extern_kernels.convolution(buf160, arg72_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg72_1
        buf162 = buf157; del buf157  # reuse
        # Topologically Sorted Source Nodes: [x_67, x_68, pow_28, mean_27], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf161, arg73_1, buf162, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf163 = buf158; del buf158  # reuse
        # Topologically Sorted Source Nodes: [x_67, x_68, pow_28, mean_27], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf162, buf163, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf166 = buf160; del buf160  # reuse
        buf164 = reinterpret_tensor(buf166, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_31], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf161, arg73_1, buf163, buf164, s0, 399360, stream=stream0)
        buf165 = reinterpret_tensor(buf166, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_67, x_68, pow_28, mean_27, add_40, sqrt_27, hidden_states_67, hidden_states_68], Original ATen: [aten.cat, aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf161, arg73_1, buf163, buf165, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del arg73_1
        del buf161
        del buf164
        del buf165
        # Topologically Sorted Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf167 = extern_kernels.convolution(buf166, arg74_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg74_1
        buf168 = buf144; del buf144  # reuse
        # Topologically Sorted Source Nodes: [x_61, x_62, x_66, output_tensor_12, x_70, output_tensor_13], Original ATen: [aten.cat, aten.convolution, aten.add]
        triton_poi_fused_add_cat_convolution_61_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_61.run(buf168, arg67_1, buf156, arg71_1, buf167, arg75_1, ps9, triton_poi_fused_add_cat_convolution_61_xnumel, stream=stream0)
        del arg67_1
        del arg71_1
        del arg75_1
        del buf156
        del buf167
        buf169 = buf162; del buf162  # reuse
        # Topologically Sorted Source Nodes: [pow_29, mean_28], Original ATen: [aten.pow, aten.mean]
        triton_red_fused_mean_pow_62_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_62.run(buf168, buf169, ps9, s0, triton_red_fused_mean_pow_62_xnumel, 128, stream=stream0)
        buf170 = buf163; del buf163  # reuse
        # Topologically Sorted Source Nodes: [pow_29, mean_28], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf169, buf170, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf173 = buf166; del buf166  # reuse
        buf171 = reinterpret_tensor(buf173, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_32], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_63.run(buf168, buf170, buf171, s0, 399360, stream=stream0)
        buf172 = reinterpret_tensor(buf173, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [pow_29, mean_28, add_42, sqrt_28, hidden_states_70, hidden_states_71], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_div_mean_pow_silu_sqrt_64_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_64.run(buf168, buf170, buf172, ps9, s0, triton_poi_fused_add_div_mean_pow_silu_sqrt_64_xnumel, stream=stream0)
        del buf171
        del buf172
        # Topologically Sorted Source Nodes: [x_72], Original ATen: [aten.convolution]
        buf174 = extern_kernels.convolution(buf173, arg76_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf174, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg76_1
        buf175 = buf169; del buf169  # reuse
        # Topologically Sorted Source Nodes: [x_72, pow_30, mean_29], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf174, arg77_1, buf175, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf176 = buf170; del buf170  # reuse
        # Topologically Sorted Source Nodes: [x_72, pow_30, mean_29], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf175, buf176, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf179 = buf173; del buf173  # reuse
        buf177 = reinterpret_tensor(buf179, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_33], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf174, arg77_1, buf176, buf177, s0, 399360, stream=stream0)
        buf178 = reinterpret_tensor(buf179, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_72, pow_30, mean_29, add_43, sqrt_29, hidden_states_72, hidden_states_73], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf174, arg77_1, buf176, buf178, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del arg77_1
        del buf174
        del buf177
        del buf178
        # Topologically Sorted Source Nodes: [x_74], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, arg78_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg78_1
        buf181 = buf175; del buf175  # reuse
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_65_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_65.run(buf168, buf180, arg79_1, buf181, ps9, s0, triton_red_fused_add_convolution_mean_pow_65_xnumel, 128, stream=stream0)
        buf182 = buf176; del buf176  # reuse
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf181, buf182, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf185 = buf179; del buf179  # reuse
        buf183 = reinterpret_tensor(buf185, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_34], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_66.run(buf168, buf180, arg79_1, buf182, buf183, s0, 399360, stream=stream0)
        buf184 = reinterpret_tensor(buf185, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, pow_31, mean_30, add_45, sqrt_30, hidden_states_75, hidden_states_76], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67.run(buf168, buf180, arg79_1, buf182, buf184, ps9, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67_xnumel, stream=stream0)
        del buf183
        del buf184
        # Topologically Sorted Source Nodes: [x_76], Original ATen: [aten.convolution]
        buf186 = extern_kernels.convolution(buf185, arg80_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg80_1
        buf187 = buf181; del buf181  # reuse
        # Topologically Sorted Source Nodes: [x_76, pow_32, mean_31], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf186, arg81_1, buf187, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf188 = buf182; del buf182  # reuse
        # Topologically Sorted Source Nodes: [x_76, pow_32, mean_31], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf187, buf188, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf191 = buf185; del buf185  # reuse
        buf189 = reinterpret_tensor(buf191, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_35], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf186, arg81_1, buf188, buf189, s0, 399360, stream=stream0)
        buf190 = reinterpret_tensor(buf191, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_76, pow_32, mean_31, add_46, sqrt_31, hidden_states_77, hidden_states_78], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf186, arg81_1, buf188, buf190, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del arg81_1
        del buf186
        del buf189
        del buf190
        # Topologically Sorted Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf191, arg82_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg82_1
        buf193 = buf168; del buf168  # reuse
        # Topologically Sorted Source Nodes: [x_74, output_tensor_14, x_78, output_tensor_15], Original ATen: [aten.convolution, aten.add]
        triton_poi_fused_add_convolution_68_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_68.run(buf193, buf180, arg79_1, buf192, arg83_1, ps9, triton_poi_fused_add_convolution_68_xnumel, stream=stream0)
        del arg79_1
        del arg83_1
        del buf180
        del buf192
        buf194 = buf187; del buf187  # reuse
        # Topologically Sorted Source Nodes: [pow_33, mean_32], Original ATen: [aten.pow, aten.mean]
        triton_red_fused_mean_pow_62_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_62.run(buf193, buf194, ps9, s0, triton_red_fused_mean_pow_62_xnumel, 128, stream=stream0)
        buf195 = buf188; del buf188  # reuse
        # Topologically Sorted Source Nodes: [pow_33, mean_32], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf194, buf195, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf198 = buf191; del buf191  # reuse
        buf196 = reinterpret_tensor(buf198, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_36], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_63.run(buf193, buf195, buf196, s0, 399360, stream=stream0)
        buf197 = reinterpret_tensor(buf198, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [pow_33, mean_32, add_48, sqrt_32, hidden_states_80, hidden_states_81], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_div_mean_pow_silu_sqrt_64_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_64.run(buf193, buf195, buf197, ps9, s0, triton_poi_fused_add_div_mean_pow_silu_sqrt_64_xnumel, stream=stream0)
        del buf196
        del buf197
        # Topologically Sorted Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf199 = extern_kernels.convolution(buf198, arg84_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg84_1
        buf200 = buf194; del buf194  # reuse
        # Topologically Sorted Source Nodes: [x_80, pow_34, mean_33], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf199, arg85_1, buf200, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf201 = buf195; del buf195  # reuse
        # Topologically Sorted Source Nodes: [x_80, pow_34, mean_33], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf200, buf201, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf204 = buf198; del buf198  # reuse
        buf202 = reinterpret_tensor(buf204, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_37], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf199, arg85_1, buf201, buf202, s0, 399360, stream=stream0)
        buf203 = reinterpret_tensor(buf204, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_80, pow_34, mean_33, add_49, sqrt_33, hidden_states_82, hidden_states_83], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf199, arg85_1, buf201, buf203, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del arg85_1
        del buf199
        del buf202
        del buf203
        # Topologically Sorted Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf204, arg86_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg86_1
        buf206 = buf200; del buf200  # reuse
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, pow_35, mean_34], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_65_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_65.run(buf193, buf205, arg87_1, buf206, ps9, s0, triton_red_fused_add_convolution_mean_pow_65_xnumel, 128, stream=stream0)
        buf207 = buf201; del buf201  # reuse
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, pow_35, mean_34], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf206, buf207, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf210 = buf204; del buf204  # reuse
        buf208 = reinterpret_tensor(buf210, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_38], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_66.run(buf193, buf205, arg87_1, buf207, buf208, s0, 399360, stream=stream0)
        buf209 = reinterpret_tensor(buf210, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, pow_35, mean_34, add_51, sqrt_34, hidden_states_85, hidden_states_86], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67.run(buf193, buf205, arg87_1, buf207, buf209, ps9, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67_xnumel, stream=stream0)
        del buf208
        del buf209
        # Topologically Sorted Source Nodes: [x_84], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, arg88_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg88_1
        buf212 = buf206; del buf206  # reuse
        # Topologically Sorted Source Nodes: [x_84, pow_36, mean_35], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf211, arg89_1, buf212, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf213 = buf207; del buf207  # reuse
        # Topologically Sorted Source Nodes: [x_84, pow_36, mean_35], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf212, buf213, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf216 = buf210; del buf210  # reuse
        buf214 = reinterpret_tensor(buf216, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_39], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf211, arg89_1, buf213, buf214, s0, 399360, stream=stream0)
        buf215 = reinterpret_tensor(buf216, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_84, pow_36, mean_35, add_52, sqrt_35, hidden_states_87, hidden_states_88], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf211, arg89_1, buf213, buf215, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del arg89_1
        del buf211
        del buf214
        del buf215
        # Topologically Sorted Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, arg90_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg90_1
        buf218 = buf193; del buf193  # reuse
        # Topologically Sorted Source Nodes: [x_82, output_tensor_16, x_86, output_tensor_17], Original ATen: [aten.convolution, aten.add]
        triton_poi_fused_add_convolution_68_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_68.run(buf218, buf205, arg87_1, buf217, arg91_1, ps9, triton_poi_fused_add_convolution_68_xnumel, stream=stream0)
        del arg87_1
        del arg91_1
        del buf205
        del buf217
        buf219 = buf212; del buf212  # reuse
        # Topologically Sorted Source Nodes: [pow_37, mean_36], Original ATen: [aten.pow, aten.mean]
        triton_red_fused_mean_pow_62_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_mean_pow_62.run(buf218, buf219, ps9, s0, triton_red_fused_mean_pow_62_xnumel, 128, stream=stream0)
        buf220 = buf213; del buf213  # reuse
        # Topologically Sorted Source Nodes: [pow_37, mean_36], Original ATen: [aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf219, buf220, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf223 = buf216; del buf216  # reuse
        buf221 = reinterpret_tensor(buf223, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_40], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_63.run(buf218, buf220, buf221, s0, 399360, stream=stream0)
        buf222 = reinterpret_tensor(buf223, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [pow_37, mean_36, add_54, sqrt_36, hidden_states_90, hidden_states_91], Original ATen: [aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_div_mean_pow_silu_sqrt_64_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_div_mean_pow_silu_sqrt_64.run(buf218, buf220, buf222, ps9, s0, triton_poi_fused_add_div_mean_pow_silu_sqrt_64_xnumel, stream=stream0)
        del buf221
        del buf222
        # Topologically Sorted Source Nodes: [x_88], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf223, arg92_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg92_1
        buf225 = buf219; del buf219  # reuse
        # Topologically Sorted Source Nodes: [x_88, pow_38, mean_37], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_red_fused_cat_convolution_mean_pow_54_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_cat_convolution_mean_pow_54.run(buf224, arg93_1, buf225, ps9, s0, triton_red_fused_cat_convolution_mean_pow_54_xnumel, 128, stream=stream0)
        buf226 = buf220; del buf220  # reuse
        # Topologically Sorted Source Nodes: [x_88, pow_38, mean_37], Original ATen: [aten.convolution, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf225, buf226, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        buf229 = buf223; del buf223  # reuse
        buf227 = reinterpret_tensor(buf229, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_41], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_56.run(buf224, arg93_1, buf226, buf227, s0, 399360, stream=stream0)
        buf228 = reinterpret_tensor(buf229, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_88, pow_38, mean_37, add_55, sqrt_37, hidden_states_92, hidden_states_93], Original ATen: [aten.convolution, aten.pow, aten.mean, aten.add, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57.run(buf224, arg93_1, buf226, buf228, ps9, s0, triton_poi_fused_add_cat_convolution_div_mean_pow_silu_sqrt_57_xnumel, stream=stream0)
        del arg93_1
        del buf224
        del buf227
        del buf228
        # Topologically Sorted Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, arg94_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (199680 + 199680*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg94_1
        buf231 = buf225; del buf225  # reuse
        # Topologically Sorted Source Nodes: [x_90, output_tensor_18, pow_39, mean_38], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_red_fused_add_convolution_mean_pow_65_xnumel = 1560 + 1560*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_red_fused_add_convolution_mean_pow_65.run(buf218, buf230, arg95_1, buf231, ps9, s0, triton_red_fused_add_convolution_mean_pow_65_xnumel, 128, stream=stream0)
        buf232 = buf226; del buf226  # reuse
        # Topologically Sorted Source Nodes: [x_90, output_tensor_18, pow_39, mean_38], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean]
        triton_per_fused_cat_convolution_mean_pow_55_xnumel = 390 + 390*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_per_fused_cat_convolution_mean_pow_55.run(buf231, buf232, s0, triton_per_fused_cat_convolution_mean_pow_55_xnumel, 4, stream=stream0)
        del buf231
        buf235 = buf229; del buf229  # reuse
        buf233 = reinterpret_tensor(buf235, (1, 512, 2, 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 0)  # alias
        # Topologically Sorted Source Nodes: [first_frame_pad_42], Original ATen: [aten.repeat]
        stream0 = get_raw_stream(0)
        triton_poi_fused_repeat_66.run(buf218, buf230, arg95_1, buf232, buf233, s0, 399360, stream=stream0)
        buf234 = reinterpret_tensor(buf235, (1, 512, 1 + (((-1) + s0) // 8), 26, 15), (599040 + 199680*(((-1) + s0) // 8), 1170 + 390*(((-1) + s0) // 8), 390, 15, 1), 780)  # alias
        # Topologically Sorted Source Nodes: [x_90, output_tensor_18, pow_39, mean_38, add_57, sqrt_38, sample, sample_1], Original ATen: [aten.convolution, aten.add, aten.pow, aten.mean, aten.sqrt, aten.div, aten.silu]
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67_xnumel = 199680 + 199680*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67.run(buf218, buf230, arg95_1, buf232, buf234, ps9, s0, triton_poi_fused_add_convolution_div_mean_pow_silu_sqrt_67_xnumel, stream=stream0)
        del arg95_1
        del buf218
        del buf230
        del buf232
        del buf233
        del buf234
        # Topologically Sorted Source Nodes: [x_92], Original ATen: [aten.convolution]
        buf236 = extern_kernels.convolution(buf235, arg96_1, stride=(1, 1, 1), padding=(0, 1, 1), dilation=(1, 1, 1), transposed=False, output_padding=(0, 0, 0), groups=1, bias=None)
        assert_size_stride(buf236, (1, 129, 1 + (((-1) + s0) // 8), 26, 15), (50310 + 50310*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1))
        del arg96_1
        del buf235
        buf237 = empty_strided_cuda((1, 256, 1 + (((-1) + s0) // 8), 26, 15), (99840 + 99840*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [sample_2], Original ATen: [aten.cat]
        triton_poi_fused_cat_69_xnumel = 99840 + 99840*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_cat_69.run(buf236, arg97_1, buf237, ps9, s0, triton_poi_fused_cat_69_xnumel, stream=stream0)
        del arg97_1
        del buf236
        buf238 = empty_strided_cuda((1, 128, 1 + (((-1) + s0) // 8), 26, 15), (49920 + 49920*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1), torch.bfloat16)
        buf239 = empty_strided_cuda((1, 128, 1 + (((-1) + s0) // 8), 26, 15), (49920 + 49920*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1), torch.bfloat16)
        buf240 = empty_strided_cuda((1, 128, 1 + (((-1) + s0) // 8), 26, 15), (49920 + 49920*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [clamp, exp_1, mul, exp], Original ATen: [aten.clamp, aten.exp, aten.mul]
        triton_poi_fused_clamp_exp_mul_70_xnumel = 49920 + 49920*(((-1) + s0) // 8)
        stream0 = get_raw_stream(0)
        triton_poi_fused_clamp_exp_mul_70.run(buf237, buf238, buf239, buf240, s0, triton_poi_fused_clamp_exp_mul_70_xnumel, stream=stream0)
    return (buf237, buf239, buf240, buf238, reinterpret_tensor(buf237, (1, 128, 1 + (((-1) + s0) // 8), 26, 15), (99840 + 99840*(((-1) + s0) // 8), 390 + 390*(((-1) + s0) // 8), 390, 15, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 9
    arg1_1 = 13178880
    arg2_1 = 9584640
    arg3_1 = rand_strided((1, 3, 9, 832, 480), (39536640, 13178880, 399360, 480, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((128, 48, 3, 3, 3), (1296, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg6_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg8_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg10_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg12_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg14_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg16_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg18_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg20_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg21_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg22_1 = rand_strided((128, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg24_1 = rand_strided((256, 128, 3, 3, 3), (3456, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg25_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg26_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg27_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg30_1 = rand_strided((256, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg32_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg34_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg35_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg36_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg37_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg38_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg40_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg42_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg44_1 = rand_strided((256, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg46_1 = rand_strided((512, 256, 3, 3, 3), (6912, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg47_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg48_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg49_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg52_1 = rand_strided((512, 256, 1, 1, 1), (256, 1, 1, 1, 1), device='cuda:0', dtype=torch.bfloat16)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg54_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg56_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg57_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg58_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg59_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg60_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg62_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg63_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg64_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg65_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg66_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg67_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg68_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg69_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg70_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg72_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg73_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg74_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg75_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg76_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg77_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg78_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg79_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg80_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg81_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg82_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg83_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg84_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg85_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg86_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg87_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg88_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg89_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg90_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg91_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg92_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg94_1 = rand_strided((512, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    arg96_1 = rand_strided((129, 512, 3, 3, 3), (13824, 27, 9, 3, 1), device='cuda:0', dtype=torch.bfloat16)
    arg97_1 = rand_strided((129, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
