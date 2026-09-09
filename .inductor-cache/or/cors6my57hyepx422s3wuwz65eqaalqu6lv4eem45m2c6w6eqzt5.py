# AOT ID: ['6_inference']
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


# kernel path: /workspace/SoulX-FlashHead/.inductor-cache/px/cpxhpqg77tyg33ibucuymqfvzbpsntr2xlo7kvmkmdguopyszi7d.py
# Topologically Sorted Source Nodes: [layer_norm, add_1, mul, add_2], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
# Source node to ATen node mapping:
#   add_1 => add_2
#   add_2 => add_3
#   layer_norm => add_1, convert_element_type, convert_element_type_1, mul, rsqrt, sub, var_mean
#   mul => mul_1
# Graph fragment:
#   %convert_element_type : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_1, torch.float32), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type, [3]), kwargs = {correction: 0, keepdim: True})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type, %getitem_3), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_2, 1e-06), kwargs = {})
#   %rsqrt : [num_users=1] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul, torch.bfloat16), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_1, 1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_1, %add_2), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %getitem), kwargs = {})
triton_red_fused_add_mul_native_layer_norm_0 = async_compile.triton('triton_red_fused_add_mul_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'in_ptr2': '*bf16', 'out_ptr2': '*bf16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=46, cc=89, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_mul_native_layer_norm_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 2, 'backend_hash': 'ED954309EC87A290DCE5A91842A64DBC742E6D6E81D7DEC7C468537DB97E535D', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_add_mul_native_layer_norm_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
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
        tmp20 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp25 = tl.load(in_ptr1 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 - tmp3
        tmp12 = 1536.0
        tmp13 = (tmp4 / tmp12)
        tmp14 = 1e-06
        tmp15 = tmp13 + tmp14
        tmp16 = libdevice.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp21 = tmp19 + tmp20
        tmp22 = 1.0
        tmp23 = tmp21 + tmp22
        tmp24 = tmp18 * tmp23
        tmp26 = tmp25 + tmp20
        tmp27 = tmp24 + tmp26
        tl.store(out_ptr2 + (r0_1 + 1536*x0), tmp27, r0_mask & xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1 = args
    args.clear()
    assert_size_stride(arg0_1, (1, 1950, 1536), (2995200, 1536, 1))
    assert_size_stride(arg1_1, (1, 1536), (1536, 1))
    assert_size_stride(arg2_1, (1, 2, 1536), (3072, 1536, 1))
    assert_size_stride(arg3_1, (128, 1536), (1536, 1))
    assert_size_stride(arg4_1, (128, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf3 = empty_strided_cuda((1, 1, 1950, 1536), (2995200, 1, 1536, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [layer_norm, add_1, mul, add_2], Original ATen: [aten.native_layer_norm, aten.add, aten.mul]
        stream0 = get_raw_stream(0)
        triton_red_fused_add_mul_native_layer_norm_0.run(arg0_1, arg2_1, arg1_1, buf3, 1950, 1536, stream=stream0)
        del arg0_1
        del arg1_1
        del arg2_1
        buf4 = empty_strided_cuda((1950, 128), (128, 1), torch.bfloat16)
        # Topologically Sorted Source Nodes: [x_1], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg4_1, reinterpret_tensor(buf3, (1950, 1536), (1536, 1), 0), reinterpret_tensor(arg3_1, (1536, 128), (1, 1536), 0), alpha=1, beta=1, out=buf4)
        del arg3_1
        del arg4_1
        del buf3
    return (reinterpret_tensor(buf4, (1, 1950, 128), (249600, 128, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((1, 1950, 1536), (2995200, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg1_1 = rand_strided((1, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg2_1 = rand_strided((1, 2, 1536), (3072, 1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((128, 1536), (1536, 1), device='cuda:0', dtype=torch.bfloat16)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
