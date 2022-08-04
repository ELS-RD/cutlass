#include <iostream>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/device_memory.h"
#include <torch/torch.h>
#include <chrono>

#include "cutlass/tensor_ref.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "helper.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/device/gemm_universal.h"


using precision_a = cutlass::half_t;
using precision_b = cutlass::half_t;
using precision_output = cutlass::half_t;
using precision_accumulator = float;
using precision_epilogue = precision_accumulator;


using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;  // cutlass::arch::OpClassSimt

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80; // cutlass::arch::Sm86

// This code section describes the tile size a thread block will compute
// thread block tile M = 128, N = 128, K = 32
using ThreadblockShape = cutlass::gemm::GemmShape<128, 128, 32>;

// This code section describes tile size a warp will compute
// warp tile M = 64, N = 64, K = 32
using WarpShape = cutlass::gemm::GemmShape<64, 64, 32>;

// MMA Op tile M = 8, N = 8, K = 4
using InstructionShape = cutlass::gemm::GemmShape<16, 8, 16>;

using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;
static int const kStages = 4;

// This is the number of elements per vectorized memory access.
// For half precision, it's 8 elements.
// This becomes the vector width of math instructions in epilogue too
static int const kEpilogueElementsPerAccess = 128 / cutlass::sizeof_bits<precision_output>::value;
using EpilogueOutputOp = cutlass::epilogue::thread::LinearCombination<
        precision_output,
        kEpilogueElementsPerAccess,
        precision_accumulator,
        precision_epilogue
>;

//using Gemm = cutlass::gemm::device::Gemm<
//        precision_a,
//        LayoutInputA,
//        precision_b,
//        LayoutInputB,
//        precision_output,
//        LayoutOutput,
//        precision_accumulator,
//        MMAOp,
//        SmArch,
//        ThreadblockShape,
//        WarpShape,
//        InstructionShape,
//        EpilogueOutputOp
//>;

using GemmU = cutlass::gemm::device::GemmUniversal<
        precision_a, LayoutInputA,
        precision_b, LayoutInputB,
        precision_output, LayoutOutput,
        float,
        MMAOp,
        SmArch,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        EpilogueOutputOp
>;

inline char const *to_string(cutlass::Status status) {

    switch (status) {
        case cutlass::Status::kSuccess:
            return "kSuccess";
        case cutlass::Status::kErrorMisalignedOperand:
            return "kErrorMisalignedOperand";
        case cutlass::Status::kErrorInvalidLayout:
            return "kErrorInvalidLayout";
        case cutlass::Status::kErrorInvalidProblem:
            return "kErrorInvalidProblem";
        case cutlass::Status::kErrorNotSupported:
            return "kErrorNotSupported";
        case cutlass::Status::kErrorWorkspaceNull:
            return "kErrorWorkspaceNull";
        case cutlass::Status::kErrorInternal:
            return "kErrorInternal";
        case cutlass::Status::kInvalid:
            return "kInvalid";
        default:
            break;
    }
    return "invalid";
}

template<typename precision, typename layout>
cutlass::TensorRef<precision, layout> toRef(torch::Tensor tensor) {
    auto leadingAxis = tensor.size(1);
    // stride is used to extract the leading dimension of each matrix, in our case it's always the nb of cols
    // as we are row oriented.
    cutlass::TensorRef<precision, layout> ref((precision *) tensor.data_ptr(), layout(leadingAxis));
    return ref;
}


//https://github.com/NVIDIA/cutlass/discussions/396 -> tuning
/// Define a CUTLASS GEMM template and launch a GEMM kernel.
void CutlassSgemmNN(
        torch::Tensor &A,
        torch::Tensor &B,
        torch::Tensor &C,
        float alpha,
        float beta) {

    int M = (int) A.size(0);
    int N = (int) B.size(1);
    int K = (int) A.size(1);

    auto ref_a = toRef<precision_a const, LayoutInputA>(A);
    auto ref_b = toRef<precision_b const, LayoutInputB>(B);
    auto ref_c = toRef<precision_output const, LayoutOutput>(C);
    auto ref_d = toRef<precision_output, LayoutOutput>(C);
//    cutlass::gemm::GemmUniversalMode mode = cutlass::gemm::GemmUniversalMode::kGemm;

//    typename Gemm::Arguments arguments{
//            {M, N, K},  // Gemm Problem dimensions
//            ref_a, // Tensor-ref for source matrix A
//            ref_b, // ... B
//            ref_c, // ... C
//            ref_d, // ... output
//            {alpha, beta}, // Scalars used in the Epilogue
//            1}; // split_k


    typename GemmU::Arguments argumentsUniversal{
            cutlass::gemm::GemmUniversalMode::kGemm,
            {M, N, K},
            1,
            {alpha, beta},
            ref_a.data(),
            ref_b.data(),
            ref_c.data(),
            ref_d.data(),
            int64_t(),
            int64_t(),
            int64_t(),
            int64_t(),
            int64_t(K),
            int64_t(N),
            int64_t(N),
            int64_t(N)
    };

    cutlass::Status status = GemmU::can_implement(argumentsUniversal);
    CHECK(status == cutlass::Status::kSuccess)
    << "arg can't be implemented by this kernel: "
    << to_string(status)
    << std::endl;


    GemmU gemm_operator;
    auto workspace_size = GemmU::get_workspace_size(argumentsUniversal);
    cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
    status = gemm_operator.initialize(argumentsUniversal, workspace.get());
    CHECK(status == cutlass::Status::kSuccess)
    << "GEMM initialization failed: "
    << to_string(status)
    << std::endl;

    status = gemm_operator();
    CHECK(status == cutlass::Status::kSuccess)
    << "GEMM execution failed: "
    << to_string(status)
    << std::endl;
}

void TestCutlassGemm(int M, int N, int K, float alpha, float beta) {
    auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .layout(torch::kStrided)
            .device(torch::kCUDA, 0)
            .requires_grad(false);

    auto tensorA = torch::rand({M, K}, options);
    auto tensorB = torch::rand({K, N}, options);
    auto tensorCCutlass = torch::empty({M, N}, options).toType(torch::kFloat16);
    auto tensorCTorchFp32 = torch::empty({M, N}, options);
    auto tensorAFp16 = tensorA.toType(torch::kFloat16);
    auto tensorBFp16 = tensorB.toType(torch::kFloat16);
    auto tensorCTorchFp16 = torch::mm(tensorAFp16, tensorBFp16);

    clock_t start, end;
    int nb_repeat = 10;
    for (int i = 0; i < nb_repeat; i++) {
        CutlassSgemmNN(tensorAFp16, tensorBFp16, tensorCCutlass, alpha, beta);
    }
    cudaDeviceSynchronize();
    start = clock();
    for (int i = 0; i < nb_repeat; i++) {
        CutlassSgemmNN(tensorAFp16, tensorBFp16, tensorCCutlass, alpha, beta);
    }
    cudaDeviceSynchronize();
    end = clock();
    auto cutlass_time = (double) (end - start) / CLOCKS_PER_SEC;
    std::cout << "cutlass time: " << cutlass_time << std::endl;

    for (int i = 0; i < nb_repeat; i++) {
        torch::mm_out(tensorCTorchFp32, tensorA, tensorB);
    }
    torch::cuda::synchronize();
    start = clock();
    for (int i = 0; i < nb_repeat; i++) {
        torch::mm_out(tensorCTorchFp32, tensorA, tensorB);
    }
    torch::cuda::synchronize();
    end = clock();
    auto torch_time = (double) (end - start) / CLOCKS_PER_SEC;
    std::cout << "torch time: " << torch_time << std::endl;
    std::cout << "speedup: " << torch_time / cutlass_time << std::endl;

    auto diffPytorchOnly = tensorCTorchFp32.sub(tensorCTorchFp16).abs().sum().item<double>();
    auto diffPytorchCutlass = tensorCTorchFp32.sub(tensorCCutlass).abs().sum().item<double>();
    auto diffDiff = abs(diffPytorchOnly - diffPytorchCutlass);

    std::cout << std::boolalpha;
    std::cout << "distance FP32 (PyTorch) - FP16 (cutlass): "
              << diffPytorchCutlass
              << std::endl;
    std::cout << "distance FP32 (PyTorch) - FP16 (PyTorch): "
              << diffPytorchOnly
              << std::endl;
    std::cout << "distance between distances: "
              << diffDiff
              << std::endl;

    CHECK(!torch::any(torch::isinf(tensorCCutlass)).item<bool>());
    CHECK(!torch::any(torch::isnan(tensorCCutlass)).item<bool>());
    CHECK(!torch::any(torch::isinf(tensorCTorchFp32)).item<bool>());
    CHECK(!torch::any(torch::isnan(tensorCTorchFp32)).item<bool>());
    CHECK(!torch::any(torch::isinf(tensorCTorchFp16)).item<bool>());
    CHECK(!torch::any(torch::isnan(tensorCTorchFp16)).item<bool>());
    CHECK(std::addressof(diffPytorchOnly) != std::addressof(diffPytorchCutlass));
    CHECK(diffPytorchOnly >=  diffPytorchCutlass);

    cudaFree(tensorA.data_ptr());
    cudaFree(tensorB.data_ptr());
    cudaFree(tensorCCutlass.data_ptr());
    cudaFree(tensorCTorchFp32.data_ptr());
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// usage:
//
//   00_basic_gemm <M> <N> <K>
//
int main(int argc, const char *arg[]) {
    torch::manual_seed(123);
    // default problem dimensions
    int problem[3] = {768, 256, 768};
    for (int i = 1; i < argc && i < 4; ++i) {
        std::stringstream ss(arg[i]);
        ss >> problem[i - 1];
    }
    std::cout << "problem size: ";
    for (const auto i: problem) {
        std::cout << i << ' ';
    }
    std::cout << std::endl;
    TestCutlassGemm(
            problem[0],
            problem[1],
            problem[2],
            1, // alpha
            0 // beta
    );

    return 0;
}
