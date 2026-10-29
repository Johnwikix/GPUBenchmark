#pragma once
#include <hip/hip_runtime_api.h>
extern "C" hipError_t gb_amd_alu_launch(int half2, unsigned accumulators, unsigned unroll,
  unsigned blocks, unsigned threads, unsigned iterations, unsigned seed, void* output,
  hipStream_t stream);
