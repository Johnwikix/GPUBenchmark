#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>
namespace cuda_bench {
struct AluConfig {int threads,workgroup,accumulators,unroll;bool fp16;};
// Both inputs and outputs contain one float per scalar lane. Half conversion
// occurs before/after the hot loop; the loop itself uses only __hfma2.
cudaError_t launch_alu(const AluConfig&,uint32_t iterations,const float* input,float* output,float multiplier,float addend0,float addend1,cudaStream_t stream);
}
