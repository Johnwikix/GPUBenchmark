#pragma once
#include <cuda_runtime.h>
#include <cstddef>
namespace cuda_bench {
int sm120_variant_count();
const char* sm120_variant_name(int variant);
cudaError_t sm120_create(int variant,int m,int n,int k,const void* a,const void* b,const void* c,void* d,float alpha,void* workspace,size_t bytes,cudaStream_t stream,void** handle);
cudaError_t sm120_launch(void* handle,cudaStream_t stream);
void sm120_destroy(void* handle);
}
