// Isolated diagnostic: reuses production allocation, validation, search and
// timers, without changing the production precision contract or published scores.
#define main production_worker_entry_unused
#include "../../workers/cuda/main.cpp"
#undef main
#include <cuda_profiler_api.h>
#include <iostream>
#include <fstream>
#include "extended_search.hpp"
#ifdef GPUBENCH_DIAG_CUPTI
#include <cupti.h>
static std::string capture_dtype;
static void CUPTIAPI trace_kernel(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void* data){
 if(domain==CUPTI_CB_DOMAIN_RESOURCE&&id==CUPTI_CBID_RESOURCE_MODULE_LOADED){
  const auto* resource=static_cast<const CUpti_ResourceData*>(data);const auto* module=static_cast<const CUpti_ModuleResourceData*>(resource->resourceDescriptor);auto dir=getenv("GPUBENCH_CAPTURE_CUBIN_DIR");
  if(dir&&module&&module->pCubin&&module->cubinSize){const auto override_token=getenv("GPUBENCH_CAPTURE_TOKEN");const std::string token=override_token?override_token:capture_dtype=="fp8"?"nvjet_sm120_qqsss_mma_128x128x64":capture_dtype=="fp16"?"cutlass_80_tensorop_s16816gemm_f16_64x256":"cutlass_80_simt_sgemm_128x256";if(std::search(module->pCubin,module->pCubin+module->cubinSize,token.begin(),token.end())!=module->pCubin+module->cubinSize){auto path=std::filesystem::path(dir)/(capture_dtype+"-module-"+std::to_string(module->moduleId)+".cubin");std::ofstream output(path,std::ios::binary);output.write(module->pCubin,module->cubinSize);std::cerr<<json{{"saved_cubin",path.string()},{"bytes",module->cubinSize}}.dump()<<'\n';}}
  return;
 }
 if(domain!=CUPTI_CB_DOMAIN_DRIVER_API)return;
 const auto* event=static_cast<const CUpti_CallbackData*>(data);
 if(domain==CUPTI_CB_DOMAIN_DRIVER_API&&event->callbackSite==CUPTI_API_ENTER&&event->symbolName&&std::string(event->functionName).find("cuLaunchKernel")==0)
  std::cerr<<json{{"actual_kernel",event->symbolName},{"launch_api",event->functionName}}.dump()<<'\n';
}
#endif
int main(int argc,char** argv){
 try{
  if(argc<4)throw std::runtime_error("cuda_gemm_probe <fp32|fp16|fp8> <size> <baseline|stable|tune|standard|nn|fastaccum> [profile|ncu]");
  const bool acc16=std::string(argv[1])=="fp16acc16";const std::string dtype=acc16?"fp16":argv[1],variant=argv[3];const uint64_t n=std::stoull(argv[2]);
  if(dtype!="fp32"&&dtype!="fp16"&&dtype!="fp8")throw std::runtime_error("only fp32, fp16 and fp8 are diagnostic inputs");
#ifdef GPUBENCH_DIAG_CUPTI
  struct Subscription{CUpti_SubscriberHandle h=nullptr;~Subscription(){if(h)cuptiUnsubscribe(h);}}subscription;
  if(argc>4&&std::string(argv[4])=="profile"){capture_dtype=dtype;auto status=cuptiSubscribe(&subscription.h,reinterpret_cast<CUpti_CallbackFunc>(trace_kernel),nullptr);if(status!=CUPTI_SUCCESS)throw std::runtime_error("CUPTI subscribe failed");cuptiEnableCallback(1,subscription.h,CUPTI_CB_DOMAIN_RESOURCE,CUPTI_CBID_RESOURCE_MODULE_LOADED);}
#endif
  SetEnvironmentVariableW(L"NVIDIA_TF32_OVERRIDE",L"0");SetEnvironmentVariableW(L"CUBLAS_EMULATE_SINGLE_PRECISION",L"0");
  cuda_check(cudaSetDevice(0),"device");LtLibrary lib;Stream stream;auto shape=make_shape(n,n,n,dtype=="fp8");json result={{"diagnostic_only",true},{"dtype",dtype},{"variant",variant},{"size",n},{"contract",acc16?"independent FP16 accumulation/output; signed dyadic exact input":dtype=="fp32"?"FP32 inputs/accumulation/output, TF32 off":"FP16/E4M3 inputs, FP32 compute/output"}};
  require_budget(gb::checked_product({n,n,24})+256ull*1024*1024,result);
  GemmBuffers buffers(shape,{dtype,20261020,0,false,acc16});GemmPlan plan(lib,shape,dtype,256ull*1024*1024,acc16);
  if(variant=="standard"){
   if(dtype!="fp32")throw std::runtime_error("standard variant only compares FP32 pedantic vs ordinary FP32");
   lib.cublasLtMatmulDescDestroy(plan.op);plan.op=nullptr;
   blas_check(lib.cublasLtMatmulDescCreate(&plan.op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"standard compute");cublasOperation_t ta=CUBLAS_OP_T,tb=CUBLAS_OP_N;
   blas_check(lib.cublasLtMatmulDescSetAttribute(plan.op,CUBLASLT_MATMUL_DESC_TRANSA,&ta,sizeof(ta)),"A transpose");blas_check(lib.cublasLtMatmulDescSetAttribute(plan.op,CUBLASLT_MATMUL_DESC_TRANSB,&tb,sizeof(tb)),"B transpose");
  }else if(variant=="nn"){
   if(dtype=="fp8")throw std::runtime_error("FP8 TN constraint retained");
   const size_t elem=buffers.spec.element_bytes();std::vector<uint8_t> packed(buffers.a.bytes);for(uint64_t row=0;row<n;row++)for(uint64_t k=0;k<n;k++){size_t index=k*n+row;float x=buffers.spec.raw(true,row,k);if(dtype=="fp32")memcpy(packed.data()+index*elem,&x,4);else{auto bits=half_encode(x);memcpy(packed.data()+index*elem,&bits,2);}}buffers.a.upload(packed.data(),packed.size());
   cublasOperation_t op=CUBLAS_OP_N;blas_check(lib.cublasLtMatmulDescSetAttribute(plan.op,CUBLASLT_MATMUL_DESC_TRANSA,&op,sizeof(op)),"NN layout");
  }else if(variant=="fastaccum"){
   if(dtype!="fp8")throw std::runtime_error("fast accumulation is an FP8-only diagnostic");int8_t on=1;blas_check(lib.cublasLtMatmulDescSetAttribute(plan.op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&on,sizeof(on)),"diagnostic fast accumulation");result["contract"]="diagnostic FP8 FAST_ACCUM=1; distinct from production accuracy mode";
  }else if(variant!="baseline"&&variant!="tune"&&variant!="stable"&&variant!="native")throw std::runtime_error("unknown variant");
  std::atomic_bool cancel=false;AlgorithmCache cache;result["search"]=select_algorithm(plan,buffers,stream.p,cancel,cache,0,variant!="baseline");result["validation"]=validate_gemm(plan,buffers,stream.p,false);result["library_version"]=lib.cublasLtGetVersion();result["build"]=gb::build_info();int driver=0,runtime=0;cudaDriverGetVersion(&driver);cudaRuntimeGetVersion(&runtime);result["cuda_versions"]={{"driver",driver},{"runtime",runtime}};
  if(variant=="native"){
#ifdef GPUBENCH_NATIVE_SM120
   result["sm120_search"]=select_native_sm120(plan,buffers,stream.p,cancel,"sm120");result["native_validation"]=validate_gemm(plan,buffers,stream.p,false);
#else
   throw std::runtime_error("native diagnostic requires pinned CUTLASS build");
#endif
  }
  result["accumulation"]=acc16?"fp16":"fp32";
  if(variant=="tune"){
   if(dtype=="fp32"||acc16)throw std::runtime_error("extended search only acc32 fp16/fp8");
   result["extended_search"]=extended_search(plan,buffers,stream.p,result["search"]);
   result["tuned_validation"]=validate_gemm(plan,buffers,stream.p,false);
  }
  if(argc>4&&(std::string(argv[4])=="profile"||std::string(argv[4])=="ncu")){
#ifdef GPUBENCH_DIAG_CUPTI
   if(subscription.h){auto status=cuptiEnableDomain(1,subscription.h,CUPTI_CB_DOMAIN_DRIVER_API);if(status!=CUPTI_SUCCESS)throw std::runtime_error("cuptiEnableDomain failed: "+std::to_string(status));}
#endif
   cuda_check(cudaProfilerStart(),"profile start");plan.execute(buffers,stream.p);cuda_check(cudaStreamSynchronize(stream.p),"profile completion");cuda_check(cudaProfilerStop(),"profile stop");result["profile_only"]=true;
#ifdef GPUBENCH_DIAG_CUPTI
   if(subscription.h)cuptiEnableDomain(0,subscription.h,CUPTI_CB_DOMAIN_DRIVER_API);
#endif
  }else{
   auto r=gb::base_result({{"id","diagnostic:cuda0"}},{{"suite","gemm"},{"dtype",dtype},{"accumulation",acc16?"fp16":"fp32"}},"cuda");r["validation_status"]="passed";Timer timer(stream.p);r=gb::measure(r,{{"samples",7},{"duration_ms",1000},{"warmup_ms",500}},cancel,gb::checked_product({2,n,n,n}),[&](uint64_t reps){return timer.time(reps,[&](uint64_t){plan.execute(buffers,stream.p);});},[](const json&){});result["measurement"]=r;result["post_validation"]=validate_gemm(plan,buffers,stream.p,false,false);
  }
  result["modules"]=gb::loaded_modules();std::cout<<result.dump(2)<<'\n';return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 4;}
}
