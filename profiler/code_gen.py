from profiler.gen_gemm import CutlassGemmProfiler


profiler: CutlassGemmProfiler = CutlassGemmProfiler(sm=80, cutlass_path="/mnt/workspace/cutlass", binary_path="/tmp/cutlass_tests")
# c.get_default(op_type="cutlass.dense", out_dtype="float16", arg0_dtype="float16", arg1_dtype="float16", use_3xtf32=True, batched=False)

result = profiler.profile(
    "cutlass.dense",
    511,
    382,
    123,
    "float16",
    "float16",
    "float16",
    use_3xtf32=False,
    profile_all_alignments=True,
    find_first_valid=False,
    use_multiprocessing=True,
    batched=False,
)

print(result)

#   // Gemm operator cutlass_tensorop_h1688gemm_64x64_32x2_tn_align8
#   using Operation_cutlass_tensorop_h1688gemm_64x64_32x2_tn_align8 = cutlass::gemm::device::Gemm<
#     cutlass::half_t, cutlass::layout::RowMajor,
#     cutlass::half_t, cutlass::layout::ColumnMajor,
#     cutlass::half_t, cutlass::layout::RowMajor,
#     cutlass::half_t,
#     cutlass::arch::OpClassTensorOp,
#     cutlass::arch::Sm75,
#     cutlass::gemm::GemmShape<64, 64, 32>,
#     cutlass::gemm::GemmShape<32, 32, 32>,
#     cutlass::gemm::GemmShape<16, 8, 8>,
#
#     cutlass::epilogue::thread::LinearCombination<
#       cutlass::half_t,
#       8,
#       cutlass::half_t,
#       cutlass::half_t
#     >,
#     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
#     2,
#     8,
#     8,
#     false,
#     cutlass::arch::OpMultiplyAdd
#   >;


#   // Gemm operator cutlass_tensorop_h1688gemm_64x64_32x2_tn_align1
#   using Operation_cutlass_tensorop_h1688gemm_64x64_32x2_tn_align1 = cutlass::gemm::device::Gemm<
#     cutlass::half_t, cutlass::layout::RowMajor,
#     cutlass::half_t, cutlass::layout::ColumnMajor,
#     cutlass::half_t, cutlass::layout::RowMajor,
#     cutlass::half_t,
#     cutlass::arch::OpClassTensorOp,
#     cutlass::arch::Sm75,
#     cutlass::gemm::GemmShape<64, 64, 32>,
#     cutlass::gemm::GemmShape<32, 32, 32>,
#     cutlass::gemm::GemmShape<16, 8, 8>,
#
#     cutlass::epilogue::thread::LinearCombination<
#       cutlass::half_t,
#       1,
#       cutlass::half_t,
#       cutlass::half_t
#     >,
#     cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
#     2,
#     1,
#     1,
#     false,
#     cutlass::arch::OpMultiplyAdd
#   >;

import torch
import time

a = torch.randn((511, 123), dtype=torch.float16, device="cuda")
b = torch.randn((123, 382), dtype=torch.float16, device="cuda")
d = torch.randn((511, 382), dtype=torch.float16, device="cuda")

for _ in range(10):
    torch.mm(a, b, out=d)
torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    torch.matmul(a, b, out=d)
    torch.cuda.synchronize()
latency = (time.time() - start) / 100
print(latency)
print(result[2])


# cd cmake-build-debug/tools/profiler/
# ./cutlass_profiler --operation=Gemm --m=511 --n=121 --k=383 --A=f16 --B=f16 --C=f16 --beta=0 --acum=fp32 --output=result.csv ---profiling-iterations=100 --warmup-iterations=10