#pragma once
#include "worker.hpp"
#include "kernels.hpp"
#include "build_metadata.hpp"
#include <filesystem>
#include <memory>
#include <cstring>
#include <sstream>
#include <iomanip>
namespace amd {
using gb::json;
inline bool device_lost=false;
inline void check(hipError_t e,const char* operation){
  if(e==hipSuccess)return;
  const std::string n=hipGetErrorName(e),m=std::string(operation)+": "+n+" "+hipGetErrorString(e);
  if(n.find("IllegalAddress")!=std::string::npos||n.find("LaunchFailure")!=std::string::npos||n.find("DeviceLost")!=std::string::npos||n.find("ContextIsDestroyed")!=std::string::npos){device_lost=true;throw gb::Error("device_lost","driver_incompatible",m);}
  if(e==hipErrorOutOfMemory)throw gb::Error("out_of_memory","configuration_unsupported",m);
  if(e==hipErrorNoBinaryForGpu||e==hipErrorInvalidDeviceFunction)throw gb::Error("kernel_missing","kernel_missing",m);
  if(e==hipErrorInsufficientDriver)throw gb::Error("driver_incompatible","driver_incompatible",m);
  throw gb::Error("hip_execution_error","probe_failed",m);
}
struct Buffer {
  void* p=nullptr;size_t bytes=0;
  explicit Buffer(size_t n):bytes(n){check(hipMalloc(&p,n),"hipMalloc");try{check(hipMemset(p,0,n),"initialize allocation");}catch(...){hipFree(p);p=nullptr;throw;}}
  ~Buffer(){if(p)hipFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
  template<class T> void upload(const std::vector<T>& x){check(hipMemcpy(p,x.data(),x.size()*sizeof(T),hipMemcpyHostToDevice),"upload");}
  std::vector<float> floats()const{std::vector<float> v(bytes/4);check(hipMemcpy(v.data(),p,bytes,hipMemcpyDeviceToHost),"readback");return v;}
};
struct Queue {
  hipStream_t stream=nullptr;hipEvent_t begin=nullptr,end=nullptr;
  Queue(){check(hipStreamCreateWithFlags(&stream,hipStreamNonBlocking),"stream create");try{check(hipEventCreate(&begin),"event create");check(hipEventCreate(&end),"event create");}catch(...){if(begin)hipEventDestroy(begin);hipStreamDestroy(stream);throw;}}
  ~Queue(){if(begin)hipEventDestroy(begin);if(end)hipEventDestroy(end);if(stream)hipStreamDestroy(stream);}
  gb::BatchTime time(uint64_t repeats,const std::function<void(uint64_t)>& launch){
    auto start=gb::Clock::now();check(hipEventRecord(begin,stream),"start event");
    for(uint64_t i=0;i<repeats;i++)launch(i);
    check(hipEventRecord(end,stream),"stop event");check(hipEventSynchronize(end),"batch completion");float ms=0;check(hipEventElapsedTime(&ms,begin,end),"elapsed event");if(!(ms>0&&std::isfinite(ms)))throw gb::Error("invalid_timing","probe_failed","HIP batch event interval is not positive and finite");return {ms,gb::millis(start,gb::Clock::now())};
  }
};
inline std::filesystem::path directory(){wchar_t p[32768];DWORD n=GetModuleFileNameW(nullptr,p,32768);if(!n||n==32768)throw gb::Error("module_path","probe_failed","cannot resolve worker directory");return std::filesystem::path(p).parent_path();}
inline std::string architecture(const json& d){auto a=d.value("architecture","");return a.substr(0,a.find(':'));}
inline bool rdna3(const std::string& a){return a=="gfx1100"||a=="gfx1101"||a=="gfx1102";}
inline bool rdna4(const std::string& a){return a=="gfx1200"||a=="gfx1201";}
inline std::string hardware(const json& d,const std::string& dtype){auto a=architecture(d);if(dtype=="fp32"||dtype=="fp16")return "supported";if(dtype=="fp8"){if(rdna4(a))return "supported";if(rdna3(a)||a.rfind("gfx10",0)==0)return "unsupported";}return "unknown";}
inline uint16_t half_encode(float x){
  uint32_t b=std::bit_cast<uint32_t>(x),s=(b>>16)&0x8000,m=b&0x7fffff;int e=int((b>>23)&255)-127+15;
  if(e>=31)return uint16_t(s|0x7c00|(m?0x200:0));
  if(e<=0){if(e< -10)return uint16_t(s);m|=0x800000;unsigned sh=unsigned(14-e),v=m>>sh,r=m&((1u<<sh)-1),h=1u<<(sh-1);return uint16_t(s+v+(r>h||(r==h&&(v&1))));}
  uint32_t v=(uint32_t(e)<<10)+(m>>13),r=m&8191;return uint16_t(s+v+(r>4096||(r==4096&&(v&1))));
}
inline float round_half(float x){return float(gb::half_decode(half_encode(x)));}
inline void finite(const std::vector<float>& values,json& r,bool requireNonzero=true){
  long double sum=0;size_t nonzero=0;for(float x:values){if(!std::isfinite(x))throw gb::Error("validation_failed","probe_failed","nonfinite full-iteration output");sum+=x;nonzero+=x!=0;}
  if(requireNonzero&&!nonzero)throw gb::Error("validation_failed","probe_failed","all output channels zero");
  r["error_metrics"]["finite_elements"]=values.size();r["error_metrics"]["checksum"]=double(sum);r["error_metrics"]["nonzero_elements"]=nonzero;
}
inline size_t budget(){size_t free=0,total=0;check(hipMemGetInfo(&free,&total),"memory budget");return free/2;}
inline unsigned groups(const json& opt){return opt.value("cache","reuse")=="rotate"?3u:1u;}
inline json effective_environment(){json values=json::object();for(const char* key:{"HIPBLASLT_LOG_LEVEL","HIPBLASLT_LOG_MASK","HIPBLASLT_LOG_FILE","HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32","ROCBLAS_LAYER"}){char value[4096]{};DWORD n=GetEnvironmentVariableA(key,value,sizeof(value));values[key]=n&&n<sizeof(value)?json(value):json(nullptr);}return values;}
inline json ready(json r){r["implementation"]="native";r["hardware_support"]="supported";r["timing_source"]="hip_events";r["timing_scope"]="device_elapsed_batch";r["build"]["device_compiler"]=gb_amd_compiler_version;r["build"]["device_compiler_path"]=gb_amd_compiler_path;r["build"]["compiled_architectures"]=gb_amd_compiled_architectures;r["build"]["hip_header_version"]=HIP_VERSION;r["configuration"]["effective_environment"]=effective_environment();return r;}
inline json failed(json r,const gb::Error& e){r["reason_code"]=e.code;r["reason_message"]=e.what();r["backend_status"]=e.status;r["result_status"]=e.status=="configuration_unsupported"||e.status=="kernel_missing"||e.status=="runtime_missing"||e.status=="api_unsupported"?"skipped":"failed";r["median_tflops"]=nullptr;r["best_tflops"]=nullptr;if(e.code=="validation_failed")r["validation_status"]="failed";return r;}
json blas_info();
json gemm(json r,const json& opt,std::atomic_bool& cancel,const gb::Progress& progress);
}
