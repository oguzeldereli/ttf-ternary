// i4gemm.cu: int4 x int4 -> int32 GEMM on the int4 tensor cores (CUTLASS 2.x SM80 kernels, run on sm_86 / sm_89),
// C [M, N] = A [M, K] B [N, K]^T with A, B packed two 4-bit codes per byte (low nibble first; K even, both contiguous
// along K). Ternary codes {-1, 0, 1} are exact in int4; activations / gradients in 4-bit codes -7..7.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"

using namespace cutlass;
using RM = layout::RowMajor; using CM = layout::ColumnMajor;
using Ep = epilogue::thread::LinearCombinationClamp<int32_t, 4, int32_t, int32_t>;
template <int TM, int TN, int TK, int WM, int WN, int ST>
using G4 = gemm::device::Gemm<int4b_t, RM, int4b_t, CM, int32_t, RM, int32_t, arch::OpClassTensorOp, arch::Sm80,
    gemm::GemmShape<TM, TN, TK>, gemm::GemmShape<WM, WN, TK>, gemm::GemmShape<16, 8, 64>, Ep,
    gemm::threadblock::GemmIdentityThreadblockSwizzle<>, ST>;

template <class G> void run(const torch::Tensor &A, const torch::Tensor &B, torch::Tensor &C, int M, int N, int K) {
  typename G::Arguments args({M, N, K}, {(int4b_t *)A.data_ptr(), K}, {(int4b_t *)B.data_ptr(), K},
                             {C.data_ptr<int32_t>(), N}, {C.data_ptr<int32_t>(), N}, {1, 0});
  G g;
  auto st = g.can_implement(args);
  TORCH_CHECK(st == Status::kSuccess, "i4gemm: cannot implement ", M, "x", N, "x", K);
  st = g.initialize(args, nullptr, at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == Status::kSuccess, "i4gemm: init failed");
  st = g(at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(st == Status::kSuccess, "i4gemm: run failed");
}

torch::Tensor i4_mm(torch::Tensor A, torch::Tensor B, int64_t cfg) {
  TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.dtype() == torch::kUInt8 && B.dtype() == torch::kUInt8);
  TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A.size(1) == B.size(1));
  int M = A.size(0), N = B.size(0), K = A.size(1) * 2;
  TORCH_CHECK(K % 64 == 0 && N % 8 == 0, "i4gemm: K % 64, N % 8");
  auto C = torch::empty({M, N}, A.options().dtype(torch::kInt32));
  switch (cfg) {
    case 0: run<G4<128, 256, 128, 64, 64, 3>>(A, B, C, M, N, K); break;
    case 1: run<G4<256, 128, 128, 64, 64, 3>>(A, B, C, M, N, K); break;
    case 2: run<G4<128, 128, 128, 64, 64, 4>>(A, B, C, M, N, K); break;
    case 3: run<G4<128, 128, 256, 64, 64, 3>>(A, B, C, M, N, K); break;
    default: run<G4<64, 128, 128, 32, 64, 4>>(A, B, C, M, N, K); break;
  }
  return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("i4_mm", &i4_mm, "int4 GEMM C = A B^T (packed, int32 out)"); }
