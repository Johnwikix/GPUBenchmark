// Instantiates NVIDIA's SM120 collective builder with E4M3 inputs and F32
// accumulator/epilogue/output. CUTLASS is pinned and licensed in the lock file.
#include "sm120_api.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/util/packed_stride.hpp>
#include <new>
static_assert(alignof(CUtensorMap)>=128,"SM120 TMA ABI requires MSVC /Zc:__cplusplus");
namespace cuda_bench {
using namespace cute;
struct Operation {virtual cudaError_t launch(cudaStream_t)=0;virtual ~Operation()=default;};
// MSVC cannot pass a 128-byte-aligned Params object by value (C2719). Keep
// NVIDIA's GEMM kernel unchanged and pass an aligned device-resident descriptor
// through this pointer-only entry point. Setup/copy stays outside event timing.
template<class Kernel> __global__ __launch_bounds__(Kernel::MaxThreadsPerBlock,Kernel::MinBlocksPerMultiprocessor)
void sm120_entry(const typename Kernel::Params* params){extern __shared__ char shared[];Kernel op;op(*params,shared);}
template<class Tile,class Schedule,bool UnitBlock=false> struct Native final:Operation {
 using E=cutlass::float_e4m3_t;using Cluster=Shape<_1,_1,_1>;
 using Epilogue=typename cutlass::epilogue::collective::CollectiveBuilder<cutlass::arch::Sm120,cutlass::arch::OpClassTensorOp,Tile,Cluster,cutlass::epilogue::collective::EpilogueTileAuto,float,float,float,cutlass::layout::ColumnMajor,4,float,cutlass::layout::ColumnMajor,4,cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;
 using Input=std::conditional_t<UnitBlock,cutlass::mx_float8_t<E>,E>;
 using OpClass=std::conditional_t<UnitBlock,cutlass::arch::OpClassBlockScaledTensorOp,cutlass::arch::OpClassTensorOp>;
 using Mainloop=typename cutlass::gemm::collective::CollectiveBuilder<cutlass::arch::Sm120,OpClass,Input,cutlass::layout::RowMajor,16,Input,cutlass::layout::ColumnMajor,16,float,Tile,Cluster,cutlass::gemm::collective::StageCountAutoCarveout<int(sizeof(typename Epilogue::SharedStorage))>,Schedule>::CollectiveOp;
 using Kernel=cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>,Mainloop,Epilogue>;
 typename Kernel::Params* device_params=nullptr;dim3 grid{},block{};
 ~Native(){if(device_params)cudaFree(device_params);}
 cudaError_t initialize(int m,int n,int k,const void* a,const void* b,const void* c,void* d,float alpha,void* workspace,size_t bytes,cudaStream_t stream,int swizzle=1){
  auto sa=cutlass::make_cute_packed_stride(typename Kernel::StrideA{},make_shape(m,k,1));
  auto sb=cutlass::make_cute_packed_stride(typename Kernel::StrideB{},make_shape(n,k,1));
  auto sc=cutlass::make_cute_packed_stride(typename Kernel::StrideC{},make_shape(m,n,1));
  auto sd=cutlass::make_cute_packed_stride(typename Kernel::StrideD{},make_shape(m,n,1));
  typename Kernel::Arguments args{};args.mode=cutlass::gemm::GemmUniversalMode::kGemm;args.problem_shape={m,n,k,1};args.scheduler.max_swizzle_size=swizzle;args.epilogue={{alpha,0.f},static_cast<const float*>(c),sc,static_cast<float*>(d),sd};
  if constexpr(UnitBlock){
   using Config=typename Mainloop::Sm1xxBlkScaledConfig;
   auto la=Config::tile_atom_to_shape_SFA(make_shape(m,n,k,1));auto lb=Config::tile_atom_to_shape_SFB(make_shape(m,n,k,1));
   args.mainloop={static_cast<const E*>(a),sa,static_cast<const E*>(b),sb,nullptr,la,nullptr,lb};
   const size_t offset=(Kernel::get_workspace_size(args)+255)/256*256,na=size_t(cosize(la)),nb=size_t(cosize(lb)),a_aligned=(na+255)/256*256;
   if(offset+a_aligned+nb>bytes)return cudaErrorMemoryAllocation;
   auto scaleA=static_cast<cutlass::float_ue8m0_t*>(static_cast<void*>(static_cast<char*>(workspace)+offset));auto scaleB=static_cast<cutlass::float_ue8m0_t*>(static_cast<void*>(static_cast<char*>(workspace)+offset+a_aligned));
   auto status=cudaMemsetAsync(scaleA,0x7f,na,stream);if(status!=cudaSuccess)return status;status=cudaMemsetAsync(scaleB,0x7f,nb,stream);if(status!=cudaSuccess)return status;
   args.mainloop={static_cast<const E*>(a),sa,static_cast<const E*>(b),sb,scaleA,la,scaleB,lb};
  }else args.mainloop={static_cast<const E*>(a),sa,static_cast<const E*>(b),sb};
  if(!Kernel::can_implement(args))return cudaErrorNotSupported;
  if(Kernel::get_workspace_size(args)>bytes)return cudaErrorMemoryAllocation;
  if(Kernel::initialize_workspace(args,workspace,stream)!=cutlass::Status::kSuccess)return cudaErrorInvalidValue;
  auto params=Kernel::to_underlying_arguments(args,workspace);grid=Kernel::get_grid_shape(params);block=Kernel::get_block_shape();
  auto status=cudaMalloc(reinterpret_cast<void**>(&device_params),sizeof(params));if(status!=cudaSuccess)return status;
  // Never publish parameters on the default stream while execution uses a
  // nonblocking stream. Complete the descriptor and unit-scale setup before
  // the host Params temporary dies and before any timed dispatch can start.
  status=cudaMemcpyAsync(device_params,&params,sizeof(params),cudaMemcpyHostToDevice,stream);if(status!=cudaSuccess)return status;
  status=cudaStreamSynchronize(stream);if(status!=cudaSuccess)return status;
  return cudaFuncSetAttribute(sm120_entry<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,Kernel::SharedStorageSize);
 }
 cudaError_t launch(cudaStream_t stream)override{sm120_entry<Kernel><<<grid,block,Kernel::SharedStorageSize,stream>>>(device_params);return cudaGetLastError();}
};
using Coop=cutlass::gemm::KernelTmaWarpSpecializedCooperative;
using Ping=cutlass::gemm::KernelTmaWarpSpecializedPingpong;
int sm120_variant_count(){return 8;}
const char* sm120_variant_name(int v){static const char* names[]={"sm120-fp8-f32-128x128x128-cooperative","sm120-fp8-f32-128x128x128-pingpong","sm120-fp8-f32-128x128x64-cooperative","sm120-fp8-f32-128x64x64-cooperative","sm120-fp8-f32-unitblock-128x128x128-cooperative","sm120-fp8-f32-unitblock-128x128x128-pingpong","sm120-fp8-f32-unitblock-128x128x128-cooperative-swizzle8","sm120-fp8-f32-unitblock-128x128x128-pingpong-swizzle8"};return v>=0&&v<8?names[v]:"invalid";}
template<class T> cudaError_t create(int m,int n,int k,const void* a,const void* b,const void* c,void* d,float alpha,void* w,size_t bytes,cudaStream_t s,void** out,int swizzle=1){auto p=new(std::nothrow) T;if(!p)return cudaErrorMemoryAllocation;auto status=p->initialize(m,n,k,a,b,c,d,alpha,w,bytes,s,swizzle);if(status!=cudaSuccess){delete p;return status;}*out=p;return cudaSuccess;}
cudaError_t sm120_create(int v,int m,int n,int k,const void* a,const void* b,const void* c,void* d,float alpha,void* w,size_t bytes,cudaStream_t s,void** out){
 *out=nullptr;
 switch(v){
 case 0:return create<Native<Shape<_128,_128,_128>,Coop>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out);
 case 1:return create<Native<Shape<_128,_128,_128>,Ping>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out);
 case 2:return create<Native<Shape<_128,_128,_64>,Coop>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out);
 case 3:return create<Native<Shape<_128,_64,_64>,Coop>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out);
 case 4:return create<Native<Shape<_128,_128,_128>,Coop,true>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out);
 case 5:return create<Native<Shape<_128,_128,_128>,Ping,true>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out);
 case 6:return create<Native<Shape<_128,_128,_128>,Coop,true>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out,8);
 case 7:return create<Native<Shape<_128,_128,_128>,Ping,true>>(m,n,k,a,b,c,d,alpha,w,bytes,s,out,8);
 default:return cudaErrorInvalidValue;
 }
}
cudaError_t sm120_launch(void* p,cudaStream_t s){return static_cast<Operation*>(p)->launch(s);}
void sm120_destroy(void* p){delete static_cast<Operation*>(p);}
}
