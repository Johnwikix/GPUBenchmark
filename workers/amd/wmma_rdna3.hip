#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1100__) && !defined(__gfx1101__) && !defined(__gfx1102__)
#error RDNA3 WMMA code object must target gfx1100, gfx1101 or gfx1102
#endif
using H16=_Float16 __attribute__((ext_vector_type(16)));
using F8=float __attribute__((ext_vector_type(8)));
// A/B: independent accumulator tiles, each 16x16 in row-major order.
// Replication between lane groups is required only on RDNA3.
template<unsigned A> __device__ void micro(const _Float16* inA,const _Float16* inB,float* out,unsigned iters){
  const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid%32,wave=tid/32;
  H16 a[A],b[A];F8 c[A];
  #pragma unroll
  for(unsigned s=0;s<A;s++){
    c[s]=F8{};
    #pragma unroll
    for(unsigned k=0;k<16;k++){a[s][k]=inA[s*256+(lane%16)*16+k];b[s][k]=inB[s*256+k*16+lane%16];}
  }
  for(unsigned i=0;i<iters;i++){
    #pragma unroll
    for(unsigned s=0;s<A;s++)c[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a[s],b[s],c[s]);
  }
  #pragma unroll
  for(unsigned s=0;s<A;s++){
    #pragma unroll
    for(unsigned g=0;g<8;g++)out[(wave*A+s)*256+(2*g+lane/16)*16+lane%16]=c[s][g];
  }
}
extern "C" __global__ void gb_wmma_fp16_a2(const _Float16* a,const _Float16* b,float* out,unsigned it){micro<2>(a,b,out,it);}
extern "C" __global__ void gb_wmma_fp16_a4(const _Float16* a,const _Float16* b,float* out,unsigned it){micro<4>(a,b,out,it);}
