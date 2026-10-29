#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "kernels.hpp"

template<unsigned A, unsigned U> __global__ void alu_float(float* out, unsigned iters, unsigned seed) {
  const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
  // Runtime arguments make the inputs opaque to constant folding. The negative
  // multiplier bounds the recurrence without a saturating/overflowing chain.
  const float mul=-0.99951171875f;
  const float bias=(1+((seed+tid)%7))*0.015625f;
  float x[A];
  #pragma unroll
  for(unsigned a=0;a<A;a++) x[a]=(1+((tid+a+seed)%31))*0.015625f;
  for(unsigned i=0;i<iters;i+=U) {
    #pragma unroll
    for(unsigned u=0;u<U;u++) {
      #pragma unroll
      for(unsigned a=0;a<A;a++) x[a]=fmaf(x[a],mul,bias+float(a)*0.001953125f);
    }
  }
  #pragma unroll
  for(unsigned a=0;a<A;a++) out[tid*A+a]=x[a];
}
template<unsigned A, unsigned U> __global__ void alu_half2(float* out, unsigned iters, unsigned seed) {
  const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x;
  __half2 x[A],b[A];
  const __half2 mul=__float2half2_rn(-0.99951171875f);
  const float bias=(1+((seed+tid)%7))*0.015625f;
  #pragma unroll
  for(unsigned a=0;a<A;a++) {
    x[a]=__floats2half2_rn((1+((tid+a+seed)%31))*0.015625f,(1+((tid+a+seed+3)%31))*0.015625f);
    b[a]=__floats2half2_rn(bias+float(a)*0.001953125f,bias+float(a+1)*0.001953125f);
  }
  for(unsigned i=0;i<iters;i+=U) {
    #pragma unroll
    for(unsigned u=0;u<U;u++) {
      #pragma unroll
      for(unsigned a=0;a<A;a++) x[a]=__hfma2(x[a],mul,b[a]);
    }
  }
  #pragma unroll
  for(unsigned a=0;a<A;a++) {
    const float2 v=__half22float2(x[a]);out[(tid*A+a)*2]=v.x;out[(tid*A+a)*2+1]=v.y;
  }
}
template<unsigned A,unsigned U> static void launch(bool h,unsigned blocks,unsigned threads,unsigned it,unsigned seed,void* out,hipStream_t stream){
  if(h)hipLaunchKernelGGL((alu_half2<A,U>),dim3(blocks),dim3(threads),0,stream,(float*)out,it,seed);
  else hipLaunchKernelGGL((alu_float<A,U>),dim3(blocks),dim3(threads),0,stream,(float*)out,it,seed);
}
extern "C" hipError_t gb_amd_alu_launch(int h,unsigned a,unsigned u,unsigned blocks,unsigned threads,unsigned it,unsigned seed,void* out,hipStream_t stream){
  if((u!=4&&u!=8) || !it || it%u)return hipErrorInvalidValue;
  if(a==4){if(u==4)launch<4,4>(h,blocks,threads,it,seed,out,stream);else launch<4,8>(h,blocks,threads,it,seed,out,stream);}
  else if(a==8){if(u==4)launch<8,4>(h,blocks,threads,it,seed,out,stream);else launch<8,8>(h,blocks,threads,it,seed,out,stream);}
  else if(a==16){if(u==4)launch<16,4>(h,blocks,threads,it,seed,out,stream);else launch<16,8>(h,blocks,threads,it,seed,out,stream);}
  else return hipErrorInvalidValue;
  return hipGetLastError();
}
