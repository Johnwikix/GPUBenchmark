#include "worker.hpp"
#include "kernel_api.hpp"
#ifdef GPUBENCH_NATIVE_SM120
#include "sm120_api.hpp"
#endif
#include <cublasLt.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <map>
#include <memory>
#include <set>
#include <sstream>

namespace {
using gb::json;
constexpr const char* cublas_reference="https://docs.nvidia.com/cuda/archive/13.2.0/cublas/index.html";
#ifdef GPUBENCH_NATIVE_SM120
constexpr const char* native_sm120_build="CUTLASS4.3.4/1810164f27f2c7a499fcfc86681392662375327d; SM120a; pointer descriptor ABI";
#else
constexpr const char* native_sm120_build="not_built";
#endif
void cuda_check(cudaError_t s,const char* where){
  if(s==cudaSuccess)return;
  std::string code="cuda_error",status="probe_failed";
  if(s==cudaErrorMemoryAllocation){code="out_of_memory";status="configuration_unsupported";}
  else if(s==cudaErrorInsufficientDriver||s==cudaErrorSystemDriverMismatch){code="driver_incompatible";status="driver_incompatible";}
  else if(s==cudaErrorNoKernelImageForDevice||s==cudaErrorInvalidDeviceFunction){code="kernel_missing";status="kernel_missing";}
  else if(s==cudaErrorNoDevice||s==cudaErrorInvalidDevice){code="device_not_found";status="configuration_unsupported";}
  else if(s==cudaErrorIllegalInstruction||s==cudaErrorMisalignedAddress||s==cudaErrorInvalidAddressSpace||s==cudaErrorInvalidPc){code="device_lost";status="probe_failed";}
  else if(s==cudaErrorLaunchFailure||s==cudaErrorLaunchTimeout||s==cudaErrorIllegalAddress||s==cudaErrorAssert){code="device_lost";status="driver_incompatible";}
  throw gb::Error(code,status,std::string(where)+": "+cudaGetErrorName(s)+": "+cudaGetErrorString(s));
}
void blas_check(cublasStatus_t s,const char* where){
  if(s==CUBLAS_STATUS_SUCCESS)return;
  std::string status=s==CUBLAS_STATUS_NOT_SUPPORTED||s==CUBLAS_STATUS_INVALID_VALUE?"configuration_unsupported":"probe_failed";
  throw gb::Error(s==CUBLAS_STATUS_NOT_SUPPORTED?"no_matching_algorithm":"cublas_error",status,std::string(where)+": cuBLAS status "+std::to_string(int(s)));
}
void check_cancel(std::atomic_bool& cancel){if(cancel)throw gb::Error("user_cancelled","probe_failed","cancelled between short batches");}
struct Buffer {
  void* p=nullptr;size_t bytes=0;
  explicit Buffer(size_t n):bytes(n){if(n)cuda_check(cudaMalloc(&p,n),"cudaMalloc");}
  ~Buffer(){if(p)cudaFree(p);}
  Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
  void upload(const void* src,size_t n){if(n>bytes)throw gb::Error("buffer_bounds","probe_failed","upload exceeds allocation");cuda_check(cudaMemcpy(p,src,n,cudaMemcpyHostToDevice),"upload");}
};
struct Stream {
  cudaStream_t p{};
  Stream(){cuda_check(cudaStreamCreateWithFlags(&p,cudaStreamNonBlocking),"create stream");}
  ~Stream(){if(p)cudaStreamDestroy(p);}
};
struct Timer {
  cudaEvent_t start{},stop{};cudaStream_t stream;
  explicit Timer(cudaStream_t s):stream(s){cuda_check(cudaEventCreate(&start),"start event");try{cuda_check(cudaEventCreate(&stop),"stop event");}catch(...){cudaEventDestroy(start);throw;}}
  ~Timer(){cudaEventDestroy(start);cudaEventDestroy(stop);}
  gb::BatchTime time(uint64_t repetitions,const std::function<void(uint64_t)>& submit){
    const auto wall=gb::Clock::now();cuda_check(cudaEventRecord(start,stream),"record start");
    for(uint64_t i=0;i<repetitions;++i)submit(i);
    cuda_check(cudaEventRecord(stop,stream),"record stop");cuda_check(cudaEventSynchronize(stop),"synchronize stop");
    float ms=0;cuda_check(cudaEventElapsedTime(&ms,start,stop),"elapsed batch");
    return {ms,gb::millis(wall,gb::Clock::now())};
  }
};
uint64_t hash64(uint64_t x){x+=0x9e3779b97f4a7c15ull;x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;return x^(x>>31);}
uint16_t half_encode(float f){
  const uint32_t bits=std::bit_cast<uint32_t>(f),sign=(bits>>16)&0x8000,exp=(bits>>23)&255,mant=bits&0x7fffff;
  if(exp==255)return uint16_t(sign|0x7c00|(mant?0x200:0));
  const int e=int(exp)-127+15;
  if(e>=31)return uint16_t(sign|0x7c00);
  if(e<=0){if(e< -10)return uint16_t(sign);const uint32_t v=mant|0x800000,shift=uint32_t(14-e),base=v>>shift,rem=v&((1u<<shift)-1),mid=1u<<(shift-1);return uint16_t(sign+base+(rem>mid||(rem==mid&&(base&1))));}
  const uint32_t base=(uint32_t(e)<<10)|(mant>>13),rem=mant&8191;
  return uint16_t(sign+base+(rem>4096||(rem==4096&&(base&1))));
}
uint8_t fp8_encode(float f){
  static const std::array<double,127> positive=[](){std::array<double,127> a{};for(size_t i=0;i<a.size();++i)a[i]=gb::e4m3_decode(uint8_t(i));return a;}();
  const double x=std::abs(double(f));auto hi=std::lower_bound(positive.begin(),positive.end(),x);
  unsigned index=hi==positive.end()?126:unsigned(hi-positive.begin());
  if(index&&x-positive[index-1]<positive[index]-x)--index;
  else if(index&&x-positive[index-1]==positive[index]-x&&((index-1)%2==0))--index;
  return uint8_t(index|(std::signbit(f)?128:0));
}
size_t byte_count(std::initializer_list<uint64_t> v){const auto n=gb::checked_product(v);if(n>SIZE_MAX)throw gb::Error("allocation_overflow","configuration_unsupported","allocation exceeds address range");return size_t(n);}
void require_budget(uint64_t bytes,json& r){size_t available=0,total=0;cuda_check(cudaMemGetInfo(&available,&total),"memory budget");r["configuration"]["available_memory_bytes"]=available;r["configuration"]["allocation_budget_bytes"]=available/2;r["configuration"]["requested_allocation_bytes"]=bytes;if(bytes>available/2)throw gb::Error("memory_budget","configuration_unsupported","requested shape and buffers exceed half of currently free device memory; choose a smaller explicit shape or workspace");}
std::string hex_bytes(const char* p,size_t n){std::ostringstream s;s<<std::hex<<std::setfill('0');for(size_t i=0;i<n;++i)s<<std::setw(2)<<unsigned(static_cast<unsigned char>(p[i]));return s.str();}
std::string hardware(const cudaDeviceProp& p,const std::string& dtype){
  const int sm=p.major*10+p.minor;
  if(dtype=="fp8"){if(sm<89)return "unsupported";if(sm==89||p.major==12)return "supported";return "unknown";}
  if(dtype=="fp32"||dtype=="fp16")return sm==75||sm==86||sm==89||p.major==12?"supported":"unknown";
  return "unknown";
}
struct LtLibrary {
  HMODULE module=nullptr;cublasLtHandle_t handle{};std::wstring path;
  #define LT_FUNCTIONS(X) X(cublasLtCreate) X(cublasLtDestroy) X(cublasLtGetVersion) X(cublasLtMatmulDescCreate) X(cublasLtMatmulDescDestroy) X(cublasLtMatmulDescSetAttribute) X(cublasLtMatrixLayoutCreate) X(cublasLtMatrixLayoutDestroy) X(cublasLtMatmulPreferenceCreate) X(cublasLtMatmulPreferenceDestroy) X(cublasLtMatmulPreferenceSetAttribute) X(cublasLtMatmulAlgoGetHeuristic) X(cublasLtMatmulAlgoCheck) X(cublasLtMatmulAlgoConfigGetAttribute) X(cublasLtMatmul)
  #define LT_MEMBER(n) decltype(&::n) n=nullptr;
  LT_FUNCTIONS(LT_MEMBER)
  #undef LT_MEMBER
  LtLibrary(){
    wchar_t exe[32768];if(!GetModuleFileNameW(nullptr,exe,32768))throw gb::Error("module_path","runtime_missing","cannot locate worker directory");
    path=(std::filesystem::path(exe).parent_path()/L"cublasLt64_13.dll").wstring();
    module=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!module)throw gb::Error("cublaslt_dll_missing","runtime_missing","private cublasLt64_13.dll or its dependency is missing; Win32 error "+std::to_string(GetLastError()));
    try{
      #define LT_RESOLVE(n) n=reinterpret_cast<decltype(n)>(GetProcAddress(module,#n));if(!n)throw gb::Error("cublaslt_symbol_missing","runtime_missing","missing symbol " #n);
      LT_FUNCTIONS(LT_RESOLVE)
      #undef LT_RESOLVE
      if(cublasLtGetVersion()!=CUBLAS_VERSION)throw gb::Error("cublaslt_version_mismatch","runtime_missing","private cuBLASLt version "+std::to_string(cublasLtGetVersion())+" does not match the locked build headers "+std::to_string(CUBLAS_VERSION));
      blas_check(cublasLtCreate(&handle),"create cuBLASLt handle");
    }catch(...){FreeLibrary(module);module=nullptr;throw;}
  }
  ~LtLibrary(){if(handle)cublasLtDestroy(handle);if(module)FreeLibrary(module);}
};
struct Shape {uint64_t m,n,k,pm,pn,pk;};
Shape make_shape(uint64_t m,uint64_t n,uint64_t k,bool pad){
  if(!m||!n||!k||m>INT_MAX-16||n>INT_MAX-16||k>INT_MAX-16)throw gb::Error("invalid_shape","configuration_unsupported","GEMM dimensions must be positive and fit the cuBLASLt integer range");
  return {m,n,k,pad?(m+15)/16*16:m,pad?(n+15)/16*16:n,pad?(k+15)/16*16:k};
}
struct DataSpec {
  std::string dtype;uint64_t seed;unsigned group;bool exact=false,acc16=false;float sa=.5f,sb=1.5f;
  float raw(bool a,uint64_t row,uint64_t col)const{
    const auto h=hash64(seed^hash64(row*0x100000001b3ull+col)^uint64_t(group)*0xf1357aea2e62a9c5ull^(a?0x169ull:0x583ull));
    if(exact)return float(int(h%17)-8)*.0625f;
    if(acc16)return h&1?.125f:-.125f; // Signed dyadic peak workload; exact reference, no relaxed error threshold.
    return (float((h>>40)&0xffffff)/16777216.f-.5f)*3.f;
  }
  double value(bool a,uint64_t row,uint64_t col)const{
    const float x=raw(a,row,col);if(dtype=="fp32")return x;if(dtype=="fp16")return gb::half_decode(half_encode(x));
    const float scale=a?sa:sb;return gb::e4m3_decode(fp8_encode(x/scale))*scale;
  }
  size_t element_bytes()const{return dtype=="fp32"?4:dtype=="fp16"?2:1;}
};
struct GemmBuffers {
  Buffer a,b,c,d,scales;DataSpec spec;Shape shape;double quant_max=0,quant_sum_square=0;uint64_t quant_count=0;
  GemmBuffers(const Shape& s,DataSpec ds):a(byte_count({s.pm,s.pk,ds.element_bytes()})),b(byte_count({s.pn,s.pk,ds.element_bytes()})),c(byte_count({s.pm,s.pn,ds.acc16?2ull:4ull})),d(c.bytes),scales(2*sizeof(float)),spec(std::move(ds)),shape(s){
    auto fill=[&](bool is_a,Buffer& dst,uint64_t outer,uint64_t logical_outer){
      std::vector<uint8_t> bytes(dst.bytes,0);
      for(uint64_t o=0;o<outer;++o)for(uint64_t z=0;z<s.pk;++z){if(o>=logical_outer||z>=s.k)continue;const uint64_t row=is_a?o:z,col=is_a?z:o,index=o*s.pk+z;const float raw=spec.raw(is_a,row,col);const double value=spec.value(is_a,row,col);const double qe=value-raw;quant_max=std::max(quant_max,std::abs(qe));quant_sum_square+=qe*qe;++quant_count;
        if(spec.dtype=="fp32")std::memcpy(bytes.data()+index*4,&raw,4);
        else if(spec.dtype=="fp16"){const auto bits=half_encode(raw);std::memcpy(bytes.data()+index*2,&bits,2);}
        else bytes[index]=fp8_encode(raw/(is_a?spec.sa:spec.sb));
      }dst.upload(bytes.data(),bytes.size());
    };
    fill(true,a,s.pm,s.m);fill(false,b,s.pn,s.n);cuda_check(cudaMemset(c.p,0,c.bytes),"zero beta input");cuda_check(cudaMemset(d.p,0,d.bytes),"zero output");const float scale[]={spec.sa,spec.sb};scales.upload(scale,sizeof(scale));
  }
};
struct GemmPlan {
  LtLibrary& lib;cublasLtMatmulDesc_t op{};cublasLtMatrixLayout_t a{},b{},c{},d{};cublasLtMatmulPreference_t preference{};cublasLtMatmulAlgo_t algorithm{};bool algorithm_ready=false;
  Shape shape;std::string dtype;bool acc16=false;Buffer workspace;float alpha=1,beta=0;uint16_t half_alpha=half_encode(1.f),half_beta=0;
  int native_variant=-1;
#ifdef GPUBENCH_NATIVE_SM120
  std::map<std::pair<int,GemmBuffers*>,void*> native_operations;
  cudaError_t prepare_native(GemmBuffers& buffers,cudaStream_t stream){
    auto key=std::make_pair(native_variant,&buffers);if(native_operations.contains(key))return cudaSuccess;
    void* op=nullptr;auto status=cuda_bench::sm120_create(native_variant,int(shape.pm),int(shape.pn),int(shape.pk),buffers.a.p,buffers.b.p,buffers.c.p,buffers.d.p,buffers.spec.sa*buffers.spec.sb,workspace.p,workspace.bytes,stream,&op);
    if(status==cudaSuccess)native_operations[key]=op;return status;
  }
#endif
  GemmPlan(LtLibrary& l,Shape s,std::string dt,size_t workspace_bytes,bool half_accumulate=false):lib(l),shape(s),dtype(std::move(dt)),acc16(half_accumulate),workspace(workspace_bytes){
    try{
      blas_check(lib.cublasLtMatmulDescCreate(&op,dtype=="fp32"?CUBLAS_COMPUTE_32F_PEDANTIC:acc16?CUBLAS_COMPUTE_16F:CUBLAS_COMPUTE_32F,acc16?CUDA_R_16F:CUDA_R_32F),"matmul descriptor");
      cublasOperation_t transa=CUBLAS_OP_T,transb=CUBLAS_OP_N;
      blas_check(lib.cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSA,&transa,sizeof(transa)),"TN A");blas_check(lib.cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_TRANSB,&transb,sizeof(transb)),"TN B");
      if(dtype=="fp8"){int8_t fast_accum=0;blas_check(lib.cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_FAST_ACCUM,&fast_accum,sizeof(fast_accum)),"disable FP8 fast accumulation");}
      const auto input_type=dtype=="fp32"?CUDA_R_32F:dtype=="fp16"?CUDA_R_16F:CUDA_R_8F_E4M3;
      blas_check(lib.cublasLtMatrixLayoutCreate(&a,input_type,s.pk,s.pm,int64_t(s.pk)),"A layout K by M");blas_check(lib.cublasLtMatrixLayoutCreate(&b,input_type,s.pk,s.pn,int64_t(s.pk)),"B layout K by N");
      blas_check(lib.cublasLtMatrixLayoutCreate(&c,acc16?CUDA_R_16F:CUDA_R_32F,s.pm,s.pn,int64_t(s.pm)),"C float layout");blas_check(lib.cublasLtMatrixLayoutCreate(&d,acc16?CUDA_R_16F:CUDA_R_32F,s.pm,s.pn,int64_t(s.pm)),"D float layout");
      blas_check(lib.cublasLtMatmulPreferenceCreate(&preference),"preference");blas_check(lib.cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspace_bytes,sizeof(workspace_bytes)),"workspace limit");
    }catch(...){release();throw;}
  }
  void release(){if(preference)lib.cublasLtMatmulPreferenceDestroy(preference);if(a)lib.cublasLtMatrixLayoutDestroy(a);if(b)lib.cublasLtMatrixLayoutDestroy(b);if(c)lib.cublasLtMatrixLayoutDestroy(c);if(d)lib.cublasLtMatrixLayoutDestroy(d);if(op)lib.cublasLtMatmulDescDestroy(op);preference=nullptr;a=b=c=d=nullptr;op=nullptr;}
  ~GemmPlan(){
#ifdef GPUBENCH_NATIVE_SM120
    for(auto& [key,op]:native_operations)cuda_bench::sm120_destroy(op);
#endif
    release();}
  void execute(GemmBuffers& buffers,cudaStream_t stream){
#ifdef GPUBENCH_NATIVE_SM120
    if(native_variant>=0){auto it=native_operations.find({native_variant,&buffers});if(it==native_operations.end())throw gb::Error("sm120_not_prepared","probe_failed","SM120 operation must be initialized before timing");cuda_check(cuda_bench::sm120_launch(it->second,stream),"SM120 native GEMM");return;}
#endif
    if(dtype=="fp8"){void* sa=buffers.scales.p;void* sb=static_cast<float*>(buffers.scales.p)+1;blas_check(lib.cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa,sizeof(sa)),"FP8 A tensor scale");blas_check(lib.cublasLtMatmulDescSetAttribute(op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb,sizeof(sb)),"FP8 B tensor scale");}
    blas_check(lib.cublasLtMatmul(lib.handle,op,acc16?static_cast<const void*>(&half_alpha):&alpha,buffers.a.p,a,buffers.b.p,b,acc16?static_cast<const void*>(&half_beta):&beta,buffers.c.p,c,buffers.d.p,d,&algorithm,workspace.p,workspace.bytes,stream),"cuBLASLt matmul");
  }
};
struct CachedAlgorithm {cublasLtMatmulAlgo_t algorithm;json search;};
using AlgorithmCache=std::map<std::string,CachedAlgorithm>;
json select_algorithm(GemmPlan& plan,GemmBuffers& buffers,cudaStream_t stream,std::atomic_bool& cancel,AlgorithmCache& cache,int device,bool steady_search=true){
  const auto& s=plan.shape;const std::string key=std::to_string(device)+":"+std::to_string(plan.lib.cublasLtGetVersion())+":"+plan.dtype+":"+std::to_string(s.pm)+":"+std::to_string(s.pn)+":"+std::to_string(s.pk)+":"+std::to_string(plan.workspace.bytes)+":"+(plan.acc16?"acc16":"acc32")+":"+(steady_search?"steady":"legacy");
  if(auto it=cache.find(key);it!=cache.end()){plan.algorithm=it->second.algorithm;plan.algorithm_ready=true;auto search=it->second.search;search["cache_hit"]=true;return search;}
  constexpr int limit=32;std::array<cublasLtMatmulHeuristicResult_t,limit> candidates{};int count=0;
  // Populate scale pointers before asking the heuristic about a scaled operation.
  if(plan.dtype=="fp8"){void* sa=buffers.scales.p;void* sb=static_cast<float*>(buffers.scales.p)+1;blas_check(plan.lib.cublasLtMatmulDescSetAttribute(plan.op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&sa,sizeof(sa)),"heuristic A scale");blas_check(plan.lib.cublasLtMatmulDescSetAttribute(plan.op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&sb,sizeof(sb)),"heuristic B scale");}
  blas_check(plan.lib.cublasLtMatmulAlgoGetHeuristic(plan.lib.handle,plan.op,plan.a,plan.b,plan.c,plan.d,plan.preference,limit,candidates.data(),&count),"algorithm heuristic");
  const bool batched=steady_search&&plan.dtype!="fp32"&&std::min({s.pm,s.pn,s.pk})>=512;
  json search={{"cache_hit",false},{"candidate_limit",limit},{"returned_candidates",count},{"tested_candidates",json::array()},{"workspace_limit_bytes",plan.workspace.bytes},{"tuning_policy",batched?"warm-device-batched-v2":"single-dispatch-v1"}};
  double best=INFINITY;Timer timer(stream);bool warmed=false;
  for(int i=0;i<count;++i){check_cancel(cancel);const auto& candidate=candidates[i];if(candidate.state!=CUBLAS_STATUS_SUCCESS||candidate.workspaceSize>plan.workspace.bytes)continue;
    cublasLtMatmulHeuristicResult_t checked{};const auto check=plan.lib.cublasLtMatmulAlgoCheck(plan.lib.handle,plan.op,plan.a,plan.b,plan.c,plan.d,&candidate.algo,&checked);if(check!=CUBLAS_STATUS_SUCCESS||checked.state!=CUBLAS_STATUS_SUCCESS)continue;
    plan.algorithm=candidate.algo;
    // cuBLAS lazily loads/JITs kernels. Initialize each candidate outside the
    // event interval before applying the per-dispatch execution budget.
    plan.execute(buffers,stream);cuda_check(cudaStreamSynchronize(stream),"candidate initialization");
    auto first=timer.time(1,[&](uint64_t){plan.execute(buffers,stream);});
    if(first.device_ms>100)throw gb::Error("dispatch_budget","configuration_unsupported","single GEMM exceeds the 100 ms conservative dispatch budget");
    if(batched&&!warmed){double elapsed=0;while(elapsed<200){check_cancel(cancel);elapsed+=timer.time(1,[&](uint64_t){plan.execute(buffers,stream);}).device_ms;}search["search_warmup_ms"]=elapsed;warmed=true;first=timer.time(1,[&](uint64_t){plan.execute(buffers,stream);});}
    const uint64_t repetitions=batched?std::clamp<uint64_t>(uint64_t(std::ceil(10/std::max(first.device_ms,.001))),1,256):1;
    std::vector<double> times;for(int t=0;t<3;++t){check_cancel(cancel);times.push_back(timer.time(repetitions,[&](uint64_t){plan.execute(buffers,stream);}).device_ms/double(repetitions));}
    const double ms=gb::median(times);int id=-1;size_t written=0;blas_check(plan.lib.cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,CUBLASLT_ALGO_CONFIG_ID,&id,sizeof(id),&written),"algorithm ID");
    json config;for(auto [name,attr]:{std::pair{"tile",CUBLASLT_ALGO_CONFIG_TILE_ID},{"stages",CUBLASLT_ALGO_CONFIG_STAGES_ID},{"split_k",CUBLASLT_ALGO_CONFIG_SPLITK_NUM},{"reduction",CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME},{"swizzle",CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING},{"custom",CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION},{"inner_shape",CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID},{"cluster_shape",CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID}}){int value=0;if(plan.lib.cublasLtMatmulAlgoConfigGetAttribute(&candidate.algo,attr,&value,sizeof(value),&written)==CUBLAS_STATUS_SUCCESS)config[name]=value;}
    search["tested_candidates"].push_back({{"heuristic_index",i},{"algorithm_id",id},{"algorithm_config",config},{"workspace_bytes",candidate.workspaceSize},{"tuning_repetitions",repetitions},{"tuning_sample_times_ms",times},{"median_ms",ms}});
    if(ms<best){best=ms;plan.algorithm_ready=true;cache[key]={candidate.algo,search};search["selected_algorithm_id"]=id;search["selected_workspace_bytes"]=candidate.workspaceSize;search["selected_config"]=config;}
  }
  if(!plan.algorithm_ready)throw gb::Error("no_matching_algorithm","configuration_unsupported","cuBLASLt returned no legal native algorithm for this dtype, TN shape and workspace");
  // The last candidate need not be the winner. Cache only the chosen opaque algo.
  const auto winner=cache.at(key).algorithm;plan.algorithm=winner;cache[key]={winner,search};return search;
}
#ifdef GPUBENCH_NATIVE_SM120
json select_native_sm120(GemmPlan& plan,GemmBuffers& buffers,cudaStream_t stream,std::atomic_bool& cancel,const std::string& policy,int forced_variant=-1){
  if(forced_variant<-1||forced_variant>=cuda_bench::sm120_variant_count())throw gb::Error("invalid_sm120_variant","configuration_unsupported","SM120 variant is outside the compiled range");
  Timer timer(stream);auto timing=[&](){std::vector<double> samples;for(int t=0;t<3;t++){check_cancel(cancel);samples.push_back(timer.time(8,[&](uint64_t){plan.execute(buffers,stream);}).device_ms/8);}return samples;};
  auto baseline=timing();const double baseline_ms=gb::median(baseline);double best=INFINITY;int winner=-1;
  json search={{"policy",policy},{"baseline_cublaslt_ms",baseline},{"tested_candidates",json::array()},{"selected_library","cuBLASLt"}};
  for(int v=forced_variant>=0?forced_variant:0;v<(forced_variant>=0?forced_variant+1:cuda_bench::sm120_variant_count());v++){
    check_cancel(cancel);plan.native_variant=v;auto status=plan.prepare_native(buffers,stream);
    if(status!=cudaSuccess){search["tested_candidates"].push_back({{"variant",cuda_bench::sm120_variant_name(v)},{"status",cudaGetErrorName(status)}});cudaGetLastError();continue;}
    plan.execute(buffers,stream);cuda_check(cudaStreamSynchronize(stream),"SM120 initialize");
    auto first=timer.time(1,[&](uint64_t){plan.execute(buffers,stream);}).device_ms;if(first>100)continue;
    auto times=timing();double ms=gb::median(times);search["tested_candidates"].push_back({{"variant",cuda_bench::sm120_variant_name(v)},{"variant_id",v},{"per_call_ms",times},{"median_ms",ms},{"status","timed"}});
    if(ms<best){best=ms;winner=v;}
  }
  const bool use_native=winner>=0&&(policy=="sm120"||best<baseline_ms*.98);plan.native_variant=use_native?winner:-1;
  // Candidates share scratch storage. Recreate the winner so its scheduler
  // and unit scale workspace are initialized after the final losing variant.
  for(auto& [key,op]:plan.native_operations)cuda_bench::sm120_destroy(op);plan.native_operations.clear();
  if(use_native)cuda_check(plan.prepare_native(buffers,stream),"SM120 winner initialization");
  if(policy=="sm120"&&!use_native)throw gb::Error("sm120_no_matching_kernel","configuration_unsupported","no legal native SM120 configuration for this explicit shape");
  if(use_native){search["selected_library"]="CUTLASS";search["selected_variant"]=cuda_bench::sm120_variant_name(winner);search["selected_variant_id"]=winner;}
  return search;
}
#endif
template<class Plan> json validate_gemm(Plan& plan,GemmBuffers& buffers,cudaStream_t stream,bool full,bool execute=true){
  if(execute)plan.execute(buffers,stream);cuda_check(cudaStreamSynchronize(stream),"validation GEMM completion");
  const auto& s=buffers.shape;std::vector<float> out(size_t(s.pm*s.pn));if(buffers.spec.acc16){std::vector<uint16_t> half(out.size());cuda_check(cudaMemcpy(half.data(),buffers.d.p,half.size()*2,cudaMemcpyDeviceToHost),"half output readback");for(size_t i=0;i<out.size();i++)out[i]=float(gb::half_decode(half[i]));}else cuda_check(cudaMemcpy(out.data(),buffers.d.p,buffers.d.bytes,cudaMemcpyDeviceToHost),"validation output readback");
  const json policy=buffers.spec.acc16?json{{"id","acc16-signed-dyadic-exact-v1"},{"K",s.k},{"absolute_floor",0},{"relative",0},{"sum_abs_factor",0}}:gb::tolerance_policy(buffers.spec.dtype,s.k);double max_abs=0,max_relative=0,max_normalized=0,checksum=0,max_unscaled_error=0;uint64_t samples=0,nonzero=0,unscaled_rejected=0;
  for(float v:out){if(!std::isfinite(v))throw gb::Error("validation_failed","probe_failed","non-finite GEMM output, including padded elements");checksum+=v;nonzero+=v!=0;}
  if(!nonzero)throw gb::Error("validation_failed","probe_failed","GEMM output is entirely zero for nonzero input");
  const uint64_t count=full?s.m*s.n:std::min<uint64_t>(128,s.m*s.n);
  for(uint64_t i=0;i<count;++i){const uint64_t index=full?i:(i<4?std::array<uint64_t,4>{0,s.n-1,(s.m-1)*s.n,s.m*s.n-1}[size_t(i)]:hash64(i^buffers.spec.seed)%(s.m*s.n));const uint64_t row=index/s.n,col=index%s.n;long double reference=0,sum_abs=0;
    for(uint64_t k=0;k<s.k;++k){const long double product=static_cast<long double>(buffers.spec.value(true,row,k))*buffers.spec.value(false,k,col);reference+=product;sum_abs+=std::abs(product);}
    const double actual=out[size_t(col*s.pm+row)],expected=double(reference),err=std::abs(actual-expected);max_abs=std::max(max_abs,err);max_relative=std::max(max_relative,err/std::max(1e-30,std::abs(expected)));max_normalized=std::max(max_normalized,err/std::max(1e-30,double(sum_abs)));++samples;
    if(buffers.spec.dtype=="fp8"){const double product=double(buffers.spec.sa)*buffers.spec.sb,unscaled=expected/product;max_unscaled_error=std::max(max_unscaled_error,std::abs(actual-unscaled));if(!gb::valid_dot(actual,unscaled,double(sum_abs)/product,policy))++unscaled_rejected;}
    if(!gb::valid_dot(actual,expected,double(sum_abs),policy))throw gb::Error("validation_failed","probe_failed","GEMM disagrees with quantized/scaled CPU reference at ("+std::to_string(row)+","+std::to_string(col)+"), actual="+std::to_string(actual)+", reference="+std::to_string(expected));
  }
  if(buffers.spec.dtype=="fp8"&&!unscaled_rejected)throw gb::Error("validation_failed","probe_failed","FP8 scale negative control was not observable under the fixed tolerance policy");
  return {{"sampled_elements",samples},{"full_matrix",full},{"checksum",checksum},{"nonzero_elements",nonzero},{"max_absolute_error",max_abs},{"max_relative_error",max_relative},{"max_error_over_sum_abs",max_normalized},{"scaled_reference_observable",buffers.spec.dtype=="fp8"?json(unscaled_rejected>0):json(nullptr)},{"unscaled_reference_rejected_elements",unscaled_rejected},{"max_unscaled_reference_error",max_unscaled_error},{"tolerance_policy",policy},{"quantization_error",{{"max_absolute",buffers.quant_max},{"rms",std::sqrt(buffers.quant_sum_square/double(buffers.quant_count))},{"elements",buffers.quant_count}}}};
}
json alu_validation(const cuda_bench::AluConfig& c,uint32_t iterations,const std::vector<float>& input,Buffer& output,bool full){
  const size_t count=size_t(c.threads)*c.accumulators*(c.fp16?2:1);std::vector<float> out(count);cuda_check(cudaMemcpy(out.data(),output.p,count*sizeof(float),cudaMemcpyDeviceToHost),"ALU validation readback");
  double checksum=0;size_t nonzero=0;for(float x:out){if(!std::isfinite(x))throw gb::Error("validation_failed","probe_failed","non-finite ALU result");checksum+=std::abs(double(x));nonzero+=x!=0;}
  if(!nonzero)throw gb::Error("validation_failed","probe_failed","ALU all-zero output");
  const size_t sampled=full?count:std::min<size_t>(64,count);double error=0;
  for(size_t s=0;s<sampled;++s){const size_t index=full?s:size_t(hash64(s)%count);float expected=input[index];const float b=c.fp16&&index%2?-.125f:.25f;
    // Runtime inputs and {-1, .25, -.125} are small dyadic fractions: every
    // recurrence is exactly representable, including both half2 lanes.
    for(uint32_t i=0;i<iterations;++i){expected=std::fma(expected,-1.f,b);if(c.fp16)expected=float(gb::half_decode(half_encode(expected)));}
    error=std::max(error,std::abs(double(out[index])-expected));if(out[index]!=expected)throw gb::Error("validation_failed","probe_failed","ALU FMA recurrence/checksum mismatch at scalar lane "+std::to_string(index));
  }
  return {{"sampled_elements",sampled},{"finite_elements",count},{"nonzero_elements",nonzero},{"absolute_checksum",checksum},{"max_absolute_error",error},{"iterations",iterations},{"full_output_reference",full}};
}

class CudaBackend final:public gb::Backend {
  std::unique_ptr<LtLibrary> lt;AlgorithmCache algorithms;std::set<int> lost;
  LtLibrary& library(){if(!lt)lt=std::make_unique<LtLibrary>();return *lt;}
  json runtime(){int driver=0,version=0;cuda_check(cudaDriverGetVersion(&driver),"CUDA driver version");cuda_check(cudaRuntimeGetVersion(&version),"CUDA runtime version");return {{"cuda_runtime",version},{"cuda_driver_api",driver},{"cuda_compile_baseline",GPUBENCH_CUDA_BASELINE},{"native_sm120_build",native_sm120_build},{"cuBLASLt",lt?json(lt->cublasLtGetVersion()):json(nullptr)},{"modules",gb::loaded_modules()},{"math_environment",{{"NVIDIA_TF32_OVERRIDE","0"},{"CUBLAS_EMULATE_SINGLE_PRECISION","0"},{"CUBLASLT_LOG_LEVEL","0"},{"CUBLAS_LOGINFO_DBG","0"}}}};}
  int select(const json& device,cudaDeviceProp& prop){const int index=device.at("backend_index").get<int>();if(lost.contains(index))throw gb::Error("device_lost","driver_incompatible","device was lost earlier; this worker will not submit further work to it");cuda_check(cudaSetDevice(index),"select CUDA device");cuda_check(cudaGetDeviceProperties(&prop,index),"device properties");return index;}
  json run_alu(json& r,const json& request,const cudaDeviceProp& prop,std::atomic_bool& cancel,const gb::Progress& progress){
    const bool half=request.value("dtype","fp32")=="fp16";const int groups=request.value("cache","reuse")=="rotate"?2:1;const int max_threads=prop.multiProcessorCount*4*256;const size_t max_count=size_t(max_threads)*16*(half?2:1);require_budget(uint64_t(max_count)*sizeof(float)*2*groups,r);
    std::vector<std::vector<float>> inputs;std::vector<std::unique_ptr<Buffer>> device_input,device_output;
    for(int g=0;g<groups;++g){auto& input=inputs.emplace_back(max_count);for(size_t i=0;i<max_count;++i){const auto h=hash64(i^request.value("seed",20261020ull)^uint64_t(g)*0x1234567);input[i]=float(5+int(h%19))*.0625f;if(h&128)input[i]=-input[i];}device_input.push_back(std::make_unique<Buffer>(max_count*sizeof(float)));device_output.push_back(std::make_unique<Buffer>(max_count*sizeof(float)));device_input.back()->upload(input.data(),input.size()*sizeof(float));}
    Stream stream;Timer timer(stream.p);cuda_bench::AluConfig selected{};uint32_t selected_iterations=0;double best_rate=0;json tuning=json::array();
    progress({{"phase","alu_tuning"}});
    for(int wg:{128,256})for(int accum:{4,8,16})for(int unroll:{4,8}){check_cancel(cancel);cuda_bench::AluConfig c{prop.multiProcessorCount*4*wg,wg,accum,unroll,half};uint32_t iterations=256;
      auto dispatch=[&](uint64_t){cuda_check(cuda_bench::launch_alu(c,iterations,static_cast<float*>(device_input[0]->p),static_cast<float*>(device_output[0]->p),-1.f,.25f,-.125f,stream.p),"CUDA ALU launch");};
      double ms=timer.time(1,dispatch).device_ms;const double wanted=std::clamp(256.*5./std::max(ms,.001),8.,1048576.);iterations=uint32_t(wanted)/uint32_t(unroll)*uint32_t(unroll);iterations=std::max(iterations,uint32_t(unroll));
      ms=timer.time(1,dispatch).device_ms;if(ms>20){iterations=std::max(uint32_t(unroll),uint32_t(double(iterations)*5./ms)/uint32_t(unroll)*uint32_t(unroll));ms=timer.time(1,dispatch).device_ms;}
      const auto ops=gb::checked_product({uint64_t(c.threads),uint64_t(accum),uint64_t(iterations),half?4ull:2ull});tuning.push_back({{"threads",c.threads},{"workgroup",wg},{"accumulators",accum},{"unroll",unroll},{"iterations",iterations},{"tuning_elapsed_ms",ms},{"flops",ops}});
      const double rate=double(ops)/ms;if(ms>0&&ms<=20&&rate>best_rate){selected=c;selected_iterations=iterations;best_rate=rate;}
    }
    if(!selected_iterations)throw gb::Error("dispatch_budget","configuration_unsupported","no ALU candidate met the 20 ms short-dispatch limit");
    json validations=json::array();for(int g=0;g<groups;++g){check_cancel(cancel);for(uint32_t iterations:{8u,selected_iterations}){cuda_check(cuda_bench::launch_alu(selected,iterations,static_cast<float*>(device_input[g]->p),static_cast<float*>(device_output[g]->p),-1.f,.25f,-.125f,stream.p),"validation ALU launch");cuda_check(cudaStreamSynchronize(stream.p),"validation ALU completion");validations.push_back(alu_validation(selected,iterations,inputs[g],*device_output[g],iterations==8));}}
    auto& cfg=r["configuration"];cfg["kernel_variant"]=half?"cuda.__hfma2":"cuda.__fmaf_rn";cfg["threads"]=selected.threads;cfg["workgroup_size"]=selected.workgroup;cfg["independent_accumulators"]=selected.accumulators;cfg["lanes_per_instruction"]=half?2:1;cfg["unroll"]=selected.unroll;cfg["iterations"]=selected_iterations;cfg["cache_policy"]=request.value("cache","reuse");cfg["buffer_groups"]=groups;cfg["tuning"]=tuning;cfg["recurrence"]={{"multiplier",-1},{"addend_lane0",.25},{"addend_lane1",-.125},{"input_generator","splitmix64-dyadic-v1"}};
    r["validation_status"]="passed";r["reference_kind"]="cpu_exact_dyadic_fma_recurrence";r["sampled_elements"]=validations.back()["sampled_elements"];r["error_metrics"]={{"pre_measurement",validations}};r["tolerance_policy"]={{"id","cuda-dyadic-fma-exact-v1"},{"absolute",0},{"finite_all_outputs",true}};
    r["native_evidence"].push_back({{"kind","documented_api_path"},{"api",half?"__hfma2":"__fmaf_rn"},{"isa_checked",false},{"instruction_evidence_required_for_certification",true}});
    uint64_t dispatch_index=0;const auto ops=gb::checked_product({uint64_t(selected.threads),uint64_t(selected.accumulators),uint64_t(selected_iterations),half?4ull:2ull});
    r=gb::measure(r,request,cancel,ops,[&](uint64_t reps){return timer.time(reps,[&](uint64_t){const size_t group=size_t(dispatch_index++%uint64_t(groups));cuda_check(cuda_bench::launch_alu(selected,selected_iterations,static_cast<float*>(device_input[group]->p),static_cast<float*>(device_output[group]->p),-1.f,.25f,-.125f,stream.p),"measured ALU launch");});},progress);
    if(r["result_status"]=="ok"){json post=json::array();for(int g=0;g<groups;++g)post.push_back(alu_validation(selected,selected_iterations,inputs[g],*device_output[g],false));r["error_metrics"]["post_measurement"]=post;}
    return r;
  }
  json run_gemm(json& r,const json& request,int device,std::atomic_bool& cancel,const gb::Progress& progress){
    auto& lib=library();const std::string dtype=request.value("dtype","fp32");const bool acc16=request.value("accumulation","fp32")=="fp16";const Shape shape=make_shape(request.value("m",1024ull),request.value("n",1024ull),request.value("k",1024ull),dtype=="fp8");const int groups=request.value("cache","reuse")=="rotate"?2:1;const size_t elem=dtype=="fp32"?4:dtype=="fp16"?2:1;
    const size_t workspace=byte_count({request.value("workspace_mib",64ull),1024,1024});const auto ab=gb::checked_product({shape.pm,shape.pk,uint64_t(elem)})+gb::checked_product({shape.pn,shape.pk,uint64_t(elem)}),cd=gb::checked_product({shape.pm,shape.pn,acc16?4ull:8ull});if(ab>UINT64_MAX-cd-8||ab+cd+8>(UINT64_MAX-workspace)/uint64_t(groups))throw gb::Error("allocation_overflow","configuration_unsupported","GEMM allocation sum overflow");require_budget((ab+cd+8)*groups+workspace,r);
    Stream stream;json validation=json::array();progress({{"phase","small_matrix_validation"}});
    // Both exact and random non-square cases are fully checked. FP8 exercises
    // zero padding, signed E4M3 values and non-unit tensor scales.
    for(bool exact:{true,false}){check_cancel(cancel);const auto small=make_shape(exact?17:31,exact?23:19,exact?19:37,dtype=="fp8");GemmBuffers buffers(small,{dtype,request.value("seed",20261020ull),0,exact,acc16});GemmPlan plan(lib,small,dtype,std::min<size_t>(workspace,16*1024*1024),acc16);select_algorithm(plan,buffers,stream.p,cancel,algorithms,device);auto v=validate_gemm(plan,buffers,stream.p,true);v["case"]=exact?"exact_signed_non_square":"random_signed_non_square";validation.push_back(v);}
    check_cancel(cancel);std::vector<std::unique_ptr<GemmBuffers>> buffers;for(int g=0;g<groups;++g)buffers.push_back(std::make_unique<GemmBuffers>(shape,DataSpec{dtype,request.value("seed",20261020ull),unsigned(g),false,acc16}));GemmPlan plan(lib,shape,dtype,workspace,acc16);progress({{"phase","gemm_algorithm_search"}});const auto search=select_algorithm(plan,*buffers[0],stream.p,cancel,algorithms,device);
    json native_search;const auto native_policy=request.value("cuda_gemm","auto");
    if(dtype=="fp8"&&native_policy!="cublaslt"){
#ifdef GPUBENCH_NATIVE_SM120
      cudaDeviceProp property{};cuda_check(cudaGetDeviceProperties(&property,device),"SM120 architecture probe");
      if(property.major==12&&property.minor==0){progress({{"phase","sm120_algorithm_search"}});native_search=select_native_sm120(plan,*buffers[0],stream.p,cancel,native_policy,request.value("sm120_variant",-1));if(plan.native_variant>=0)for(auto& buffer:buffers)cuda_check(plan.prepare_native(*buffer,stream.p),"SM120 buffer-group initialization");}
      else if(native_policy=="sm120")throw gb::Error("sm120_architecture_required","configuration_unsupported","native SM120 GEMM requires compute capability12.0");
#else
      if(native_policy=="sm120")throw gb::Error("sm120_kernel_not_built","kernel_missing","rebuild CUDA worker with locked CUTLASS4.3.4 to enable native SM120 GEMM");
#endif
    }

    progress({{"phase","performance_matrix_validation"},{"native_variant",plan.native_variant}});
    for(int g=0;g<groups;++g){check_cancel(cancel);validation.push_back(validate_gemm(plan,*buffers[g],stream.p,shape.m*shape.n<=4096));}
#ifdef GPUBENCH_NATIVE_SM120
    if(plan.native_variant>=0)for(bool exact:{true,false}){
      progress({{"phase","native_small_validation"},{"exact",exact},{"native_variant",plan.native_variant}});
      check_cancel(cancel);auto small=make_shape(exact?17:31,exact?23:19,exact?19:37,true);if(plan.native_variant>=4){small.pm=128;small.pn=128;small.pk=128;}GemmBuffers input(small,{dtype,request.value("seed",20261020ull),0,exact});GemmPlan test(lib,small,dtype,std::min<size_t>(workspace,16*1024*1024));test.native_variant=plan.native_variant;cuda_check(test.prepare_native(input,stream.p),"native small matrix preparation");auto v=validate_gemm(test,input,stream.p,true);v["case"]=exact?"native_exact_signed_non_square":"native_random_signed_non_square";validation.push_back(v);
    }
#endif
    auto& cfg=r["configuration"];cfg["logical_shape"]={{"m",shape.m},{"n",shape.n},{"k",shape.k}};cfg["physical_shape"]={{"m",shape.pm},{"n",shape.pn},{"k",shape.pk}};cfg["layout"]="column_major_TN; physical A[K,M], B[K,N], C/D[M,N]";cfg["lda"]=shape.pk;cfg["ldb"]=shape.pk;cfg["ldc"]=shape.pm;cfg["alpha"]=1;cfg["beta"]=0;cfg["compute_type"]=dtype=="fp32"?"CUBLAS_COMPUTE_32F_PEDANTIC":acc16?"CUBLAS_COMPUTE_16F":"CUBLAS_COMPUTE_32F";cfg["scale_type"]=acc16?"CUDA_R_16F":"CUDA_R_32F";cfg["fast_accum"]=false;cfg["tf32"]=false;cfg["single_precision_emulation"]=false;cfg["workspace_bytes"]=workspace;cfg["algorithm_search"]=search;cfg["algorithm_id"]=search["selected_algorithm_id"];cfg["cache_policy"]=request.value("cache","reuse");cfg["buffer_groups"]=groups;cfg["library"]="cuBLASLt";cfg["library_version"]=lib.cublasLtGetVersion();cfg["input_generator"]=acc16?"splitmix64-signed-dyadic-acc16-v1":"splitmix64-signed-v1";if(dtype=="fp8")cfg["tensor_scales"]={{"a",buffers[0]->spec.sa},{"b",buffers[0]->spec.sb},{"mode","CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F (documented default)"}};
    if(!native_search.is_null())cfg["sm120_search"]=native_search;
    if(plan.native_variant>=0){cfg["library"]="CUTLASS";cfg["library_version"]="4.3.4";cfg["algorithm_id"]=nullptr;cfg["compute_type"]="native_sm120_mma_f32";cfg["scale_application"]="FP32 epilogue alpha=scaleA*scaleB";cfg["native_epilogue_alpha"]=buffers[0]->spec.sa*buffers[0]->spec.sb;cfg["kernel_variant"]=native_search["selected_variant"];cfg["internal_block_scales"]=plan.native_variant>=4?json{{"type","UE8M0"},{"value",1.0},{"encoding",127},{"source","all unit values prepared before timing; actual per-tensor scales remain0.5/1.5"}}:json(nullptr);cfg["fast_accum_control"]="native F32 MMA accumulator; no cuBLAS fast-accum mode";}
    r["validation_status"]="passed";r["reference_kind"]="cpu_long_double_dot_actual_quantized_scaled_inputs";r["sampled_elements"]=validation[2]["sampled_elements"];r["tolerance_policy"]=validation[2]["tolerance_policy"];r["error_metrics"]={{"pre_measurement",validation}};r["native_evidence"].push_back({{"kind","documented_api_path"},{"api",plan.native_variant>=0?"NVIDIA CUTLASS Sm120 GemmUniversal/CollectiveMma":"cublasLtMatmul"},{"reference",plan.native_variant>=0?"https://github.com/NVIDIA/cutlass/tree/1810164f27f2c7a499fcfc86681392662375327d":cublas_reference},{"compute_type",cfg["compute_type"]},{"profiler_checked",false}});
    Timer timer(stream.p);uint64_t index=0;const auto ops=gb::checked_product({2,shape.m,shape.n,shape.k});
    r["calibration_linearity"]=json::array();double single_ms=0;
    for(uint64_t repetitions:{1ull,2ull,4ull,8ull}){check_cancel(cancel);if(single_ms*double(repetitions)>200){r["calibration_linearity"].push_back({{"repetitions",repetitions},{"status","skipped"},{"reason","would exceed 200 ms short-batch budget"}});continue;}
      const auto time=timer.time(repetitions,[&](uint64_t){plan.execute(*buffers[size_t(index++%uint64_t(groups))],stream.p);});if(repetitions==1)single_ms=time.device_ms;
      r["calibration_linearity"].push_back({{"repetitions",repetitions},{"elapsed_ms",time.device_ms},{"wall_ms",time.wall_ms},{"flops",gb::checked_product({ops,repetitions})}});
    }
    r=gb::measure(r,request,cancel,ops,[&](uint64_t reps){return timer.time(reps,[&](uint64_t){plan.execute(*buffers[size_t(index++%uint64_t(groups))],stream.p);});},progress);
    if(r["result_status"]=="ok"){json post=json::array();for(auto& buffer:buffers)post.push_back(validate_gemm(plan,*buffer,stream.p,false,false));r["error_metrics"]["post_measurement"]=post;}
    return r;
  }
public:
  std::string name()const override{return "cuda";}
  json enumerate()override{
    int count=0;const auto status=cudaGetDeviceCount(&count);if(status==cudaErrorNoDevice)return json::array();cuda_check(status,"CUDA enumerate");json devices=json::array();
    for(int i=0;i<count;++i){cudaDeviceProp p{};cuda_check(cudaGetDeviceProperties(&p,i),"CUDA properties");char pci[64]{};cuda_check(cudaDeviceGetPCIBusId(pci,sizeof(pci),i),"CUDA PCI bus ID");const auto uuid=hex_bytes(p.uuid.bytes,16);json identity={{"uuid",uuid},{"pci_bus_id",pci},{"luid",hex_bytes(p.luid,8)},{"luid_node_mask",p.luidDeviceNodeMask}};
      devices.push_back({{"id","cuda:"+uuid},{"alias","nvidia:"+std::to_string(i)},{"backend_index",i},{"name",p.name},{"vendor_id",0x10de},{"device_id",nullptr},{"architecture","sm_"+std::to_string(p.major)+std::to_string(p.minor)},{"memory_bytes",p.totalGlobalMem},{"integrated",bool(p.integrated)},{"identity",identity},{"compute_capability",{{"major",p.major},{"minor",p.minor}}},{"multiprocessor_count",p.multiProcessorCount},{"best_effort",std::string(p.name).find("GeForce")==std::string::npos}});
    }return devices;
  }
  json probe(const json& request)override{
    cudaDeviceProp prop{};select(request.at("device"),prop);json cases=json::array();std::atomic_bool cancel=false;gb::Progress progress=[](const json&){};
    for(const std::string dtype:{"fp32","fp16","fp8"})for(const std::string suite:{"alu","gemm"}){if(suite=="alu"&&dtype=="fp8")continue;json item={{"case_id",gb::case_id(suite,dtype)},{"hardware_support",hardware(prop,dtype)},{"backend_status","ready"},{"probe_execution","not_run"}};
      if(hardware(prop,dtype)=="unsupported"){item["backend_status"]="api_unsupported";item["reason_code"]="hardware_unsupported";cases.push_back(item);continue;}
      try{Stream stream;if(suite=="alu"){cuda_bench::AluConfig c{128,128,4,4,dtype=="fp16"};const size_t count=128*4*(c.fp16?2:1);std::vector<float> input(count,.3125f);Buffer a(count*4),b(count*4);a.upload(input.data(),input.size()*4);cuda_check(cuda_bench::launch_alu(c,8,static_cast<float*>(a.p),static_cast<float*>(b.p),-1,.25f,-.125f,stream.p),"probe ALU");cuda_check(cudaStreamSynchronize(stream.p),"probe ALU completion");item["validation"]=alu_validation(c,8,input,b,true);}
        else{const auto shape=make_shape(17,23,19,dtype=="fp8");GemmBuffers buffers(shape,{dtype,20261020,0,true});GemmPlan plan(library(),shape,dtype,16*1024*1024);select_algorithm(plan,buffers,stream.p,cancel,algorithms,request["device"]["backend_index"]);item["validation"]=validate_gemm(plan,buffers,stream.p,true);}
        item["probe_execution"]="passed";item["hardware_support"]="supported";
      }catch(const gb::Error& e){item["backend_status"]=e.status;item["reason_code"]=e.code;item["reason_message"]=e.what();item["probe_execution"]="failed";}
      cases.push_back(item);
    }
    json half_case={{"case_id","gemm.fp16.acc16"},{"hardware_support",hardware(prop,"fp16")},{"backend_status","ready"}};
    try{Stream stream;auto shape=make_shape(17,23,19,false);GemmBuffers input(shape,{"fp16",20261020,0,true,true});GemmPlan plan(library(),shape,"fp16",16*1024*1024,true);select_algorithm(plan,input,stream.p,cancel,algorithms,request["device"]["backend_index"]);half_case["validation"]=validate_gemm(plan,input,stream.p,true);half_case["probe_execution"]="passed";}
    catch(const gb::Error& e){half_case["probe_execution"]="failed";half_case["backend_status"]=e.status;half_case["reason_code"]=e.code;half_case["reason_message"]=e.what();}
    cases.push_back(half_case);return {{"cases",cases},{"runtime",runtime()},{"hardware_certification","consult_project_evidence_for_this_device_and_build"},{"timing_source","cuda_events"},{"timing_scope","device_elapsed_batch"}};
  }
  json run(const json& request,std::atomic_bool& cancel,const gb::Progress& progress)override{
    json r=gb::base_result(request.at("device"),request,"cuda");int device=-1;
    try{cudaDeviceProp prop{};device=select(request.at("device"),prop);const auto suite=request.value("suite","alu"),dtype=request.value("dtype","fp32");if(request.value("accumulation","fp32")!="fp32"&&!(suite=="gemm"&&dtype=="fp16"&&request["accumulation"]=="fp16"))return gb::unavailable(r,"unknown","api_unsupported","accumulation_not_implemented","only FP16 GEMM has a separate FP16 accumulation case");r["hardware_support"]=hardware(prop,dtype);r["build"]["runtime"]=runtime();
      if(dtype!="fp32"&&dtype!="fp16"&&dtype!="fp8")return gb::unavailable(r,"unknown","api_unsupported","dtype_not_implemented","CUDA worker accepts only fp32, fp16 and fp8");
      if(hardware(prop,dtype)=="unsupported")return gb::unavailable(r,"unsupported","api_unsupported","hardware_unsupported","OCP FP8 is unavailable before Ada SM89 on the targeted GeForce architectures");
      if(suite!="alu"&&suite!="gemm")return gb::unavailable(r,r["hardware_support"],"api_unsupported","suite_not_implemented","CUDA implements ALU and full dense GEMM");
      if(suite=="alu"&&dtype=="fp8")return gb::unavailable(r,r["hardware_support"],"api_unsupported","alu_dtype_not_implemented","ALU contract includes FP32 and FP16 only");
      r["implementation"]="native";r["timing_source"]="cuda_events";r["timing_scope"]="device_elapsed_batch";
      r=suite=="alu"?run_alu(r,request,prop,cancel,progress):run_gemm(r,request,device,cancel,progress);if(r.value("reason_code","")=="device_lost")lost.insert(device);r["build"]["runtime"]=runtime();return r;
    }catch(const gb::Error& e){if(e.code=="device_lost"&&device>=0)lost.insert(device);r["reason_code"]=e.code;r["reason_message"]=e.what();r["backend_status"]=e.status;r["result_status"]=e.code=="user_cancelled"?"cancelled":e.status=="configuration_unsupported"||e.status=="runtime_missing"||e.status=="kernel_missing"||e.status=="api_unsupported"?"skipped":"failed";if(e.code=="validation_failed")r["validation_status"]="failed";r["median_tflops"]=nullptr;r["best_tflops"]=nullptr;return r;}
  }
};
}
int main(){
  // Set before any runtime/library initialization, including standalone use.
  SetEnvironmentVariableW(L"NVIDIA_TF32_OVERRIDE",L"0");SetEnvironmentVariableW(L"CUBLAS_EMULATE_SINGLE_PRECISION",L"0");SetEnvironmentVariableW(L"CUBLASLT_LOG_LEVEL",L"0");SetEnvironmentVariableW(L"CUBLAS_LOGINFO_DBG",L"0");SetEnvironmentVariableW(L"CUBLAS_LOGTRACE_DBG",L"0");
  CudaBackend backend;return gb::worker_main(backend);
}
