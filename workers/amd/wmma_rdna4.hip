#include <hip/hip_runtime.h>
#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1200__) && !defined(__gfx1201__)
#error RDNA4 WMMA code object must target gfx1200 or gfx1201
#endif
using H8=_Float16 __attribute__((ext_vector_type(8)));
using I2=int __attribute__((ext_vector_type(2)));
using F8=float __attribute__((ext_vector_type(8)));
// FP16 distributes K in groups of four between the lane groups. FP8 uses
// contiguous groups of eight instead. AMD's matrix instruction calculator
// defines these separate mappings; do not reuse the FP16 map for FP8.
__device__ unsigned k_index(unsigned lane,unsigned j){return (j/4)*8+(lane/16)*4+j%4;}
template<unsigned A> __device__ void micro16(const _Float16* inA,const _Float16* inB,float* out,unsigned iters){
  const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid%32,wave=tid/32;
  H8 a[A],b[A];F8 c[A];
  #pragma unroll
  for(unsigned s=0;s<A;s++){
    c[s]=F8{};
    #pragma unroll
    for(unsigned j=0;j<8;j++){unsigned k=k_index(lane,j);a[s][j]=inA[s*256+(lane%16)*16+k];b[s][j]=inB[s*256+k*16+lane%16];}
  }
  for(unsigned i=0;i<iters;i++){
    #pragma unroll
    for(unsigned s=0;s<A;s++)c[s]=__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[s],b[s],c[s]);
  }
  #pragma unroll
  for(unsigned s=0;s<A;s++){
    #pragma unroll
    for(unsigned g=0;g<8;g++)out[(wave*A+s)*256+(lane/16*8+g)*16+lane%16]=c[s][g];
  }
}
template<unsigned A> __device__ void micro8(const unsigned char* inA,const unsigned char* inB,float* out,unsigned iters){
  const unsigned tid=blockIdx.x*blockDim.x+threadIdx.x,lane=tid%32,wave=tid/32;
  I2 a[A],b[A];F8 c[A];
  #pragma unroll
  for(unsigned s=0;s<A;s++){
    a[s]=I2{};b[s]=I2{};c[s]=F8{};
    #pragma unroll
    for(unsigned j=0;j<8;j++){
      unsigned k=(lane/16)*8+j;
      a[s][j/4]|=unsigned(inA[s*256+(lane%16)*16+k])<<(8*(j%4));
      b[s][j/4]|=unsigned(inB[s*256+k*16+lane%16])<<(8*(j%4));
    }
  }
  for(unsigned i=0;i<iters;i++){
    #pragma unroll
    for(unsigned s=0;s<A;s++)c[s]=__builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a[s],b[s],c[s]);
  }
  #pragma unroll
  for(unsigned s=0;s<A;s++){
    #pragma unroll
    for(unsigned g=0;g<8;g++)out[(wave*A+s)*256+(lane/16*8+g)*16+lane%16]=c[s][g];
  }
}
extern "C" __global__ void gb_wmma_fp16_a2(const _Float16* a,const _Float16* b,float* out,unsigned it){micro16<2>(a,b,out,it);}
extern "C" __global__ void gb_wmma_fp16_a4(const _Float16* a,const _Float16* b,float* out,unsigned it){micro16<4>(a,b,out,it);}
extern "C" __global__ void gb_wmma_fp8_a2(const unsigned char* a,const unsigned char* b,float* out,unsigned it){micro8<2>(a,b,out,it);}
extern "C" __global__ void gb_wmma_fp8_a4(const unsigned char* a,const unsigned char* b,float* out,unsigned it){micro8<4>(a,b,out,it);}
