#include "kernel_api.hpp"
#include <cuda_fp16.h>
namespace cuda_bench {
template<int Acc,int Unroll> __global__ void alu32(uint32_t iterations,const float* input,float* output,float multiplier,float addend){
  const unsigned t=blockIdx.x*blockDim.x+threadIdx.x;
  float x[Acc];
  #pragma unroll
  for(int a=0;a<Acc;++a)x[a]=input[t*Acc+a];
  for(uint32_t it=0;it<iterations;it+=Unroll){
    #pragma unroll
    for(int u=0;u<Unroll;++u){
      #pragma unroll
      for(int a=0;a<Acc;++a)x[a]=__fmaf_rn(x[a],multiplier,addend);
    }
  }
  #pragma unroll
  for(int a=0;a<Acc;++a)output[t*Acc+a]=x[a];
}
template<int Acc,int Unroll> __global__ void alu16(uint32_t iterations,const float* input,float* output,float multiplier,float addend0,float addend1){
  const unsigned t=blockIdx.x*blockDim.x+threadIdx.x;
  const __half2 m=__float2half2_rn(multiplier),b=__floats2half2_rn(addend0,addend1);
  __half2 x[Acc];
  #pragma unroll
  for(int a=0;a<Acc;++a)x[a]=__floats2half2_rn(input[(t*Acc+a)*2],input[(t*Acc+a)*2+1]);
  for(uint32_t it=0;it<iterations;it+=Unroll){
    #pragma unroll
    for(int u=0;u<Unroll;++u){
      #pragma unroll
      for(int a=0;a<Acc;++a)x[a]=__hfma2(x[a],m,b);
    }
  }
  #pragma unroll
  for(int a=0;a<Acc;++a){const float2 y=__half22float2(x[a]);output[(t*Acc+a)*2]=y.x;output[(t*Acc+a)*2+1]=y.y;}
}
template<int Acc,int Unroll> void dispatch(const AluConfig& c,uint32_t it,const float* input,float* output,float m,float b0,float b1,cudaStream_t stream){
  if(c.fp16)alu16<Acc,Unroll><<<c.threads/c.workgroup,c.workgroup,0,stream>>>(it,input,output,m,b0,b1);
  else alu32<Acc,Unroll><<<c.threads/c.workgroup,c.workgroup,0,stream>>>(it,input,output,m,b0);
}
cudaError_t launch_alu(const AluConfig& c,uint32_t it,const float* input,float* output,float m,float b0,float b1,cudaStream_t stream){
  if(it==0||it%c.unroll||c.threads%c.workgroup)return cudaErrorInvalidValue;
  if(c.accumulators==4){if(c.unroll==4)dispatch<4,4>(c,it,input,output,m,b0,b1,stream);else dispatch<4,8>(c,it,input,output,m,b0,b1,stream);}
  else if(c.accumulators==8){if(c.unroll==4)dispatch<8,4>(c,it,input,output,m,b0,b1,stream);else dispatch<8,8>(c,it,input,output,m,b0,b1,stream);}
  else if(c.accumulators==16){if(c.unroll==4)dispatch<16,4>(c,it,input,output,m,b0,b1,stream);else dispatch<16,8>(c,it,input,output,m,b0,b1,stream);}
  else return cudaErrorInvalidValue;
  return cudaGetLastError();
}
}
