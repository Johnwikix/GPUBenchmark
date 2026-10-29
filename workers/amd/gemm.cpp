#include "amd.hpp"
#ifdef GB_HAVE_HIPBLASLT
#include <hipblaslt/hipblaslt.h>
#endif
#ifdef GB_HAVE_ROCBLAS
#include <rocblas/rocblas.h>
#endif
namespace amd {
struct Library {
  HMODULE module=nullptr;json diagnostic;
  Library(const char* library,std::initializer_list<const wchar_t*> filenames){
    diagnostic={{"library",library},{"backend_status","runtime_missing"},{"attempted_paths",json::array()}};
    for(auto file:filenames){auto path=directory()/file;diagnostic["attempted_paths"].push_back(path.string());if(!std::filesystem::is_regular_file(path))continue;
      module=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
      if(module){wchar_t actual[32768];GetModuleFileNameW(module,actual,32768);diagnostic["loaded_path"]=std::filesystem::path(actual).string();diagnostic["backend_status"]="ready";break;}diagnostic["windows_error"]=GetLastError();
    }
  }
  ~Library(){if(module)FreeLibrary(module);}
  template<class F> F symbol(const char* name){auto p=module?GetProcAddress(module,name):nullptr;if(!p)throw gb::Error("library_symbol_missing","runtime_missing",std::string("optional BLAS export absent: ")+name);return reinterpret_cast<F>(p);}
};
json blas_info(){
  json info=json::array();
  Library lt("hipBLASLt",{L"hipblaslt.dll",L"hipblaslt_1.dll",L"hipblaslt_0.dll"}),rb("rocBLAS",{L"rocblas.dll",L"rocblas_5.dll",L"rocblas_4.dll"});
#ifndef GB_HAVE_HIPBLASLT
  lt.diagnostic["backend_status"]="api_unsupported";lt.diagnostic["reason_message"]="worker built without hipBLASLt headers";
#endif
#ifndef GB_HAVE_ROCBLAS
  rb.diagnostic["backend_status"]="api_unsupported";rb.diagnostic["reason_message"]="worker built without rocBLAS headers";
#endif
  info.push_back(lt.diagnostic);info.push_back(rb.diagnostic);return info;
}
struct Inputs {
  uint64_t m,n,k,allocatedBytes=0;unsigned groupCount;std::string dtype;double scaleA,scaleB;
  std::vector<double> a,b;double quantA=0,quantB=0;
  struct Group {std::unique_ptr<Buffer> a,b,d;};std::vector<Group> groups;
  explicit Inputs(const json& opt):m(opt.value("m",1024ull)),n(opt.value("n",1024ull)),k(opt.value("k",1024ull)),groupCount(amd::groups(opt)),dtype(opt.value("dtype","fp32")),scaleA(dtype=="fp8"?.5:1.),scaleB(dtype=="fp8"?1.5:1.){
    if(!m||!n||!k||m>INT_MAX||n>INT_MAX||k>INT_MAX)throw gb::Error("invalid_shape","configuration_unsupported","positive GEMM dimensions within int32 range required");
    const unsigned elementBytes=dtype=="fp32"?4:dtype=="fp16"?2:1;const uint64_t na=gb::checked_product({m,k}),nb=gb::checked_product({k,n}),nd=gb::checked_product({m,n});
    uint64_t need=gb::checked_product({groupCount,gb::checked_product({na+nb,elementBytes})+gb::checked_product({nd,4})});
    if(need>budget())throw gb::Error("memory_budget","configuration_unsupported","explicit GEMM shape exceeds half the currently free GPU memory; choose a smaller shape");allocatedBytes=need;
    a.resize(size_t(na));b.resize(size_t(nb));std::vector<uint8_t> bytesA(size_t(na)*elementBytes),bytesB(size_t(nb)*elementBytes);
    auto fill=[&](std::vector<double>& reference,std::vector<uint8_t>& bytes,double scale,unsigned salt,double& quant){
      std::mt19937 rng(opt.value("seed",20261020u)+salt);
      for(size_t i=0;i<reference.size();i++){
        float value=(int(rng()%129)-64)*.015625f;
        if(dtype=="fp32"){reference[i]=value;std::memcpy(bytes.data()+i*4,&value,4);}
        else if(dtype=="fp16"){uint16_t h=half_encode(value);reference[i]=gb::half_decode(h);std::memcpy(bytes.data()+i*2,&h,2);}
        else{uint8_t v=gb::e4m3_encode(value/scale);reference[i]=gb::e4m3_decode(v)*scale;bytes[i]=v;}
        quant=std::max(quant,std::abs(reference[i]-value));
      }
    };
    fill(a,bytesA,scaleA,0,quantA);fill(b,bytesB,scaleB,7919,quantB);
    for(unsigned g=0;g<groupCount;g++){
      Group group{std::make_unique<Buffer>(bytesA.size()),std::make_unique<Buffer>(bytesB.size()),std::make_unique<Buffer>(size_t(nd)*4)};
      // Physical A is column-major KxM with op(A)=transpose; physical B is
      // column-major KxN. Logical reference A is row-major MxK, B is column-major.
      group.a->upload(bytesA);group.b->upload(bytesB);check(hipMemset(group.d->p,0,group.d->bytes),"clear C/D");groups.push_back(std::move(group));
    }
  }
  void validate(unsigned group,json& r){
    const auto actual=groups[group].d->floats();finite(actual,r,false);const uint64_t count=m*n;
    const uint64_t sampled=count<=4096?count:std::min<uint64_t>(256,count);double maxAbs=0,maxNormalized=0;auto policy=gb::tolerance_policy(dtype,k);
    for(uint64_t sample=0;sample<sampled;sample++){
      uint64_t index=count<=4096?sample:(sample*(count-1)/(sampled-1)),row=index%m,col=index/m;long double expected=0,sumAbs=0;
      for(uint64_t z=0;z<k;z++){double product=a[row*k+z]*b[col*k+z];expected+=product;sumAbs+=std::abs(product);}
      const double error=std::abs(actual[index]-double(expected));maxAbs=std::max(maxAbs,error);maxNormalized=std::max(maxNormalized,error/std::max(1e-12,double(sumAbs)));
      if(!gb::valid_dot(actual[index],double(expected),double(sumAbs),policy))throw gb::Error("validation_failed","probe_failed","GEMM CPU reference mismatch at row="+std::to_string(row)+", col="+std::to_string(col));
    }
    r["validation_status"]="passed";r["reference_kind"]=count<=4096?"cpu_long_double_complete":"cpu_long_double_sampled";r["sampled_elements"]=sampled;r["tolerance_policy"]=policy;
    r["error_metrics"]["max_abs_error"]=maxAbs;r["error_metrics"]["max_normalized_error"]=maxNormalized;r["error_metrics"]["quantization_error"]={{"A_max_abs",quantA},{"B_max_abs",quantB}};
  }
  void metadata(json& r){r["configuration"].update({{"logical_shape",{m,n,k}},{"physical_shape",{m,n,k}},{"layout","column_major_TN_A_KxM_B_KxN_D_MxN"},{"alpha",1},{"beta",0},{"buffer_groups",groupCount},{"allocated_matrix_bytes",allocatedBytes},{"input_scale_a",scaleA},{"input_scale_b",scaleB}});}
};
#ifdef GB_HAVE_HIPBLASLT
// decltype retains the SDK's exact calling convention and opaque ABI. Taking
// function types does not create import-table references to optional DLLs.
#define LT_EXPORTS(X) X(hipblasLtCreate) X(hipblasLtDestroy) X(hipblasLtGetVersion) X(hipblasLtMatmulDescCreate) X(hipblasLtMatmulDescDestroy) X(hipblasLtMatmulDescSetAttribute) X(hipblasLtMatrixLayoutCreate) X(hipblasLtMatrixLayoutDestroy) X(hipblasLtMatmulPreferenceCreate) X(hipblasLtMatmulPreferenceDestroy) X(hipblasLtMatmulPreferenceSetAttribute) X(hipblasLtMatmulAlgoGetHeuristic) X(hipblasLtMatmul)
struct LtApi {
  Library library{"hipBLASLt",{L"hipblaslt.dll",L"hipblaslt_1.dll",L"hipblaslt_0.dll"}};
#define DECLARE(name) decltype(&::name) name=nullptr;
  LT_EXPORTS(DECLARE)
#undef DECLARE
  LtApi(){if(!library.module)throw gb::Error("hipblaslt_missing","runtime_missing","hipBLASLt DLL absent or dependency load failed beside amd_worker.exe; include matching kernel data");
#define RESOLVE(name) name=library.symbol<decltype(name)>(#name);
    LT_EXPORTS(RESOLVE)
#undef RESOLVE
  }
};
void ltcheck(hipblasStatus_t status,const char* operation){if(status==HIPBLAS_STATUS_SUCCESS)return;std::string code="hipblaslt_error",backend="probe_failed";if(status==HIPBLAS_STATUS_NOT_SUPPORTED||status==HIPBLAS_STATUS_ARCH_MISMATCH){code="no_matching_algorithm";backend="kernel_missing";}else if(status==HIPBLAS_STATUS_INVALID_VALUE){code="library_configuration";backend="configuration_unsupported";}else if(status==HIPBLAS_STATUS_ALLOC_FAILED){code="library_out_of_memory";backend="configuration_unsupported";}throw gb::Error(code,backend,std::string(operation)+" status="+std::to_string(int(status)));}
struct LtPlan {
  LtApi api;hipblasLtHandle_t handle=nullptr;hipblasLtMatmulDesc_t desc=nullptr;hipblasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;hipblasLtMatmulPreference_t pref=nullptr;
  ~LtPlan(){if(pref)api.hipblasLtMatmulPreferenceDestroy(pref);if(a)api.hipblasLtMatrixLayoutDestroy(a);if(b)api.hipblasLtMatrixLayoutDestroy(b);if(c)api.hipblasLtMatrixLayoutDestroy(c);if(desc)api.hipblasLtMatmulDescDestroy(desc);if(handle)api.hipblasLtDestroy(handle);}
};
json lt_gemm(json r,const json& opt,std::atomic_bool& cancel,const gb::Progress& progress){
  LtPlan plan;ltcheck(plan.api.hipblasLtCreate(&plan.handle),"hipblasLtCreate");const size_t memoryLimit=budget();Inputs data(opt);Queue q;
  hipDataType type=data.dtype=="fp32"?HIP_R_32F:data.dtype=="fp16"?HIP_R_16F:HIP_R_8F_E4M3;
  ltcheck(plan.api.hipblasLtMatmulDescCreate(&plan.desc,HIPBLAS_COMPUTE_32F,HIP_R_32F),"matmul descriptor");
  hipblasOperation_t transA=HIPBLAS_OP_T,transB=HIPBLAS_OP_N;
  ltcheck(plan.api.hipblasLtMatmulDescSetAttribute(plan.desc,HIPBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)),"transpose A");
  ltcheck(plan.api.hipblasLtMatmulDescSetAttribute(plan.desc,HIPBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)),"transpose B");
  ltcheck(plan.api.hipblasLtMatrixLayoutCreate(&plan.a,type,data.k,data.m,int64_t(data.k)),"layout A");
  ltcheck(plan.api.hipblasLtMatrixLayoutCreate(&plan.b,type,data.k,data.n,int64_t(data.k)),"layout B");
  ltcheck(plan.api.hipblasLtMatrixLayoutCreate(&plan.c,HIP_R_32F,data.m,data.n,int64_t(data.m)),"layout C/D");
  Buffer scales(8);std::vector<float> hostScales{float(data.scaleA),float(data.scaleB)};scales.upload(hostScales);
  if(data.dtype=="fp8"){
    void* a=scales.p;void* b=static_cast<char*>(scales.p)+4;
    hipblasLtMatmulMatrixScale_t mode=HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    ltcheck(plan.api.hipblasLtMatmulDescSetAttribute(plan.desc,HIPBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)),"A tensor scale mode");
    ltcheck(plan.api.hipblasLtMatmulDescSetAttribute(plan.desc,HIPBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)),"B tensor scale mode");
    ltcheck(plan.api.hipblasLtMatmulDescSetAttribute(plan.desc,HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER,&a,sizeof(a)),"A tensor scale");
    ltcheck(plan.api.hipblasLtMatmulDescSetAttribute(plan.desc,HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER,&b,sizeof(b)),"B tensor scale");
  }
  uint64_t requestedWorkspace=gb::checked_product({opt.value("workspace_mib",64ull),1024,1024});const uint64_t available=memoryLimit>data.allocatedBytes+8?memoryLimit-data.allocatedBytes-8:0;size_t workspaceBytes=size_t(std::min({requestedWorkspace,available,uint64_t(budget())}));
  std::unique_ptr<Buffer> workspace;if(workspaceBytes)workspace=std::make_unique<Buffer>(workspaceBytes);
  ltcheck(plan.api.hipblasLtMatmulPreferenceCreate(&plan.pref),"preference create");
  ltcheck(plan.api.hipblasLtMatmulPreferenceSetAttribute(plan.pref,HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspaceBytes,sizeof(workspaceBytes)),"workspace preference");
  hipblasLtMatmulHeuristicResult_t candidates[16]{};int found=0;
  ltcheck(plan.api.hipblasLtMatmulAlgoGetHeuristic(plan.handle,plan.desc,plan.a,plan.b,plan.c,plan.c,plan.pref,16,candidates,&found),"algorithm heuristic");
  if(!found)throw gb::Error("no_matching_algorithm","kernel_missing","hipBLASLt found no native algorithm; inspect matching kernel data, version, TN layout, dimensions, workspace; hardware capability unchanged");
  float alpha=1,beta=0;int selected=-1;double fastest=INFINITY;json tuning=json::array();
  auto invoke=[&](int algorithm,uint64_t index){auto& group=data.groups[index%data.groupCount];return plan.api.hipblasLtMatmul(plan.handle,plan.desc,&alpha,group.a->p,plan.a,group.b->p,plan.b,&beta,group.d->p,plan.c,group.d->p,plan.c,&candidates[algorithm].algo,workspace?workspace->p:nullptr,workspaceBytes,q.stream);};
  for(int i=0;i<found;i++){
    if(cancel)break;if(candidates[i].state!=HIPBLAS_STATUS_SUCCESS||candidates[i].workspaceSize>workspaceBytes)continue;
    auto status=invoke(i,0);if(status!=HIPBLAS_STATUS_SUCCESS){tuning.push_back({{"candidate",i},{"library_status",int(status)}});continue;}check(hipStreamSynchronize(q.stream),"candidate warmup");
    auto elapsed=q.time(1,[&](uint64_t index){ltcheck(invoke(i,index),"candidate GEMM");});
    if(elapsed.device_ms>100)throw gb::Error("dispatch_budget","configuration_unsupported","single GEMM exceeds 100 ms; select a smaller explicit shape");
    data.validate(0,r);tuning.push_back({{"candidate",i},{"workspace_bytes",candidates[i].workspaceSize},{"elapsed_ms",elapsed.device_ms},{"validation_status","passed"}});
    if(elapsed.device_ms<fastest){fastest=elapsed.device_ms;selected=i;}
  }
  if(cancel){r["result_status"]="cancelled";r["reason_code"]="user_cancelled";return r;}
  if(selected<0)throw gb::Error("no_matching_algorithm","kernel_missing","all hipBLASLt algorithm candidates rejected; no precision fallback used");
  // Validate all rotated buffers with the selected algorithm before sampling.
  for(unsigned g=0;g<data.groupCount;g++){q.time(1,[&](uint64_t){ltcheck(invoke(selected,g),"selected GEMM validation");});data.validate(g,r);}
  int version=0;ltcheck(plan.api.hipblasLtGetVersion(plan.handle,&version),"library version");
  r=ready(r);data.metadata(r);r["configuration"].update({{"library","hipBLASLt"},{"library_version",version},{"library_path",plan.api.library.diagnostic["loaded_path"]},{"compute_type","HIPBLAS_COMPUTE_32F"},{"compute_fast_tf32",false},{"compute_fast_bf16",false},{"workspace_limit_bytes",requestedWorkspace},{"workspace_bytes",workspaceBytes},{"tuning_candidates",tuning},{"algorithm_candidate_index",selected}});
  const auto* opaque=reinterpret_cast<const unsigned char*>(&candidates[selected].algo);std::ostringstream encoded;for(size_t i=0;i<sizeof(candidates[selected].algo);i++)encoded<<std::hex<<std::setw(2)<<std::setfill('0')<<unsigned(opaque[i]);r["configuration"]["algorithm_id"]="opaque-sdk-abi:"+encoded.str();
  r["native_evidence"].push_back({{"kind","documented_api_path"},{"detail","hipBLASLt explicit FP32 compute, no FAST_TF32/BF16, original input types; kernel ISA certification pending"}});
  uint64_t cursor=0;const auto ops=gb::checked_product({2,data.m,data.n,data.k});
  r=gb::measure(r,opt,cancel,ops,[&](uint64_t repeats){return q.time(repeats,[&](uint64_t){ltcheck(invoke(selected,cursor++),"timed GEMM");});},progress);
  try{for(unsigned g=0;g<data.groupCount;g++)data.validate(g,r);}catch(const gb::Error& e){return failed(r,e);}r["build"]["loaded_modules"]=gb::loaded_modules();return r;
}
#endif
#ifdef GB_HAVE_ROCBLAS
#define RB_EXPORTS(X) X(rocblas_create_handle) X(rocblas_destroy_handle) X(rocblas_set_stream) X(rocblas_set_math_mode) X(rocblas_set_pointer_mode) X(rocblas_sgemm) X(rocblas_get_version_string)
struct RbApi {
  Library library{"rocBLAS",{L"rocblas.dll",L"rocblas_5.dll",L"rocblas_4.dll"}};
#define DECLARE(name) decltype(&::name) name=nullptr;
  RB_EXPORTS(DECLARE)
#undef DECLARE
  RbApi(){if(!library.module)throw gb::Error("rocblas_missing","runtime_missing","rocBLAS strict FP32 fallback DLL absent or cannot load beside amd_worker.exe");
#define RESOLVE(name) name=library.symbol<decltype(name)>(#name);
    RB_EXPORTS(RESOLVE)
#undef RESOLVE
  }
};
void rbcheck(rocblas_status status,const char* operation){if(status==rocblas_status_success)return;throw gb::Error("rocblas_error",status==rocblas_status_not_implemented?"kernel_missing":status==rocblas_status_invalid_size||status==rocblas_status_memory_error?"configuration_unsupported":"probe_failed",std::string(operation)+" status="+std::to_string(int(status)));}
struct RbPlan {RbApi api;rocblas_handle handle=nullptr;~RbPlan(){if(handle)api.rocblas_destroy_handle(handle);}};
json rb_gemm(json r,const json& opt,std::atomic_bool& cancel,const gb::Progress& progress){
  RbPlan plan;rbcheck(plan.api.rocblas_create_handle(&plan.handle),"rocBLAS create");Inputs data(opt);Queue q;
  rbcheck(plan.api.rocblas_set_stream(plan.handle,q.stream),"rocBLAS stream");rbcheck(plan.api.rocblas_set_math_mode(plan.handle,rocblas_default_math),"rocBLAS strict default math");rbcheck(plan.api.rocblas_set_pointer_mode(plan.handle,rocblas_pointer_mode_host),"rocBLAS host scalars");float alpha=1,beta=0;
  auto invoke=[&](uint64_t index){auto& g=data.groups[index%data.groupCount];rbcheck(plan.api.rocblas_sgemm(plan.handle,rocblas_operation_transpose,rocblas_operation_none,int(data.m),int(data.n),int(data.k),&alpha,(const float*)g.a->p,int(data.k),(const float*)g.b->p,int(data.k),&beta,(float*)g.d->p,int(data.m)),"rocBLAS SGEMM");};
  for(unsigned g=0;g<data.groupCount;g++){auto t=q.time(1,[&](uint64_t){invoke(g);});if(t.device_ms>100)throw gb::Error("dispatch_budget","configuration_unsupported","rocBLAS SGEMM exceeds 100 ms");data.validate(g,r);}
  char version[128]{};rbcheck(plan.api.rocblas_get_version_string(version,sizeof(version)),"rocBLAS version");r=ready(r);data.metadata(r);
  r["configuration"].update({{"library","rocBLAS"},{"library_version",version},{"library_path",plan.api.library.diagnostic["loaded_path"]},{"algorithm_id","rocblas_sgemm-default"},{"math_mode","rocblas_default_math"},{"xf32_enabled",false},{"workspace_bytes",0}});
  r["native_evidence"].push_back({{"kind","documented_api_path"},{"detail","rocBLAS SGEMM FP32 inputs/output with explicit rocblas_default_math; no XF32 mode"}});
  uint64_t cursor=0;r=gb::measure(r,opt,cancel,gb::checked_product({2,data.m,data.n,data.k}),[&](uint64_t repeats){return q.time(repeats,[&](uint64_t){invoke(cursor++);});},progress);
  try{for(unsigned g=0;g<data.groupCount;g++)data.validate(g,r);}catch(const gb::Error& e){return failed(r,e);}r["build"]["loaded_modules"]=gb::loaded_modules();return r;
}
#endif
json gemm(json r,const json& opt,std::atomic_bool& cancel,const gb::Progress& progress){
  gb::Error original("hipblaslt_headers_missing","api_unsupported","worker built without hipBLASLt headers; rebuild with ROCm Core SDK math headers");
#ifdef GB_HAVE_HIPBLASLT
  try{return lt_gemm(r,opt,cancel,progress);}catch(const gb::Error& e){
    // Correctness/async/device failures must not be hidden by another library.
    if(e.status!="runtime_missing"&&e.status!="kernel_missing"&&e.status!="configuration_unsupported")throw;
    original=e;
  }
#endif
  if(opt.value("dtype","fp32")=="fp32"){
#ifdef GB_HAVE_ROCBLAS
    try{r["configuration"]["hipblaslt_fallback_reason"]={{"code",original.code},{"message",original.what()}};return rb_gemm(r,opt,cancel,progress);}catch(const gb::Error& e){r["configuration"]["rocblas_fallback_error"]={{"code",e.code},{"message",e.what()}};return failed(r,e);}
#else
    r["configuration"]["rocblas_fallback_error"]="worker built without rocBLAS headers";
#endif
  }
  return failed(r,original);
}
}
