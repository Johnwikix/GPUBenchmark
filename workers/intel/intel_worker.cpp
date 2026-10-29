// Native Windows Intel GPU worker. No SYCL default selector or CPU fallback.
#define MKL_BLAS_COMPUTE_MODE oneapi::mkl::blas::compute_mode::standard
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <mkl_version.h>
#include <mkl_service.h>
#include "worker.hpp"
#include <dxgi1_4.h>
#include <wrl/client.h>
#include <array>
#include <cctype>
#include <cstring>
#include <iomanip>
#include <memory>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <type_traits>
#if __INTEL_MKL__ != 2026
#error "This worker targets oneMKL 2026; update and validate the dependency lock before changing SDK series"
#endif

namespace intel {
using gb::json;
using Microsoft::WRL::ComPtr;
constexpr auto gemm_source = "https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/gemm.html";
constexpr auto mode_source = "https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2026-0/compute-modes.html";
constexpr auto info_source = "https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_intel_device_info.asciidoc";

std::string lower(std::string s) {
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c){ return char(std::tolower(c)); });
  return s;
}
template<class Range> std::string hex_bytes(const Range& bytes) {
  std::ostringstream s; s << std::hex << std::setfill('0');
  for(auto b: bytes) s << std::setw(2) << unsigned(b);
  return s.str();
}
std::string backend_name(const sycl::device& d) {
  if(d.get_backend()==sycl::backend::ext_oneapi_level_zero)return "level_zero";
  if(d.get_backend()==sycl::backend::opencl)return "opencl";
  return "sycl_backend_"+std::to_string(static_cast<int>(d.get_backend()));
}
uint64_t add_bytes(uint64_t a,uint64_t b) {
  if(a>UINT64_MAX-b)throw gb::Error("count_overflow","configuration_unsupported","Allocation byte sum overflow");
  return a+b;
}
json runtime_version() {
  MKLVersion v{};mkl_get_version(&v);
  return {{"major",v.MajorVersion},{"minor",v.MinorVersion},{"update",v.UpdateVersion},
    {"build",v.Build?v.Build:""},{"product_status",v.ProductStatus?v.ProductStatus:""},
    {"platform",v.Platform?v.Platform:""},{"processor",v.Processor?v.Processor:""}};
}
bool has_fma(const sycl::device& d, bool half) {
  auto v = half ? d.get_info<sycl::info::device::half_fp_config>()
                : d.get_info<sycl::info::device::single_fp_config>();
  return std::find(v.begin(),v.end(),sycl::info::fp_config::fma)!=v.end() &&
         std::find(v.begin(),v.end(),sycl::info::fp_config::soft_float)==v.end();
}
json fp_config(const sycl::device& d, bool half) {
  json out=json::array();
  for(auto v: half ? d.get_info<sycl::info::device::half_fp_config>()
                   : d.get_info<sycl::info::device::single_fp_config>()) {
    const char* name="other";
    switch(v) {
      case sycl::info::fp_config::denorm: name="denorm";break;
      case sycl::info::fp_config::inf_nan: name="inf_nan";break;
      case sycl::info::fp_config::round_to_nearest: name="round_to_nearest";break;
      case sycl::info::fp_config::round_to_zero: name="round_to_zero";break;
      case sycl::info::fp_config::round_to_inf: name="round_to_inf";break;
      case sycl::info::fp_config::fma: name="fma";break;
      case sycl::info::fp_config::soft_float: name="soft_float";break;
      default: break;
    }
    out.push_back(name);
  }
  return out;
}
std::string architecture(const std::string& name) {
  static const std::regex a(R"(Arc.*\bA[357][0-9]{2}M?\b)",std::regex::icase);
  static const std::regex b(R"(Arc.*\bB[567][0-9]{2}M?\b)",std::regex::icase);
  if(std::regex_search(name,a))return "arc_a";
  if(std::regex_search(name,b))return "arc_b";
  return "unknown";
}
bool excluded_product(const std::string& name) {
  auto s=lower(name);
  static const std::regex arc_pro(R"(Arc.*\bPro\b)",std::regex::icase);
  return std::regex_search(name,arc_pro) || s.find("flex")!=std::string::npos ||
         s.find("max ")!=std::string::npos || s.find("data center")!=std::string::npos ||
         s.find("data-center")!=std::string::npos;
}
json describe(const sycl::device& d, size_t index) {
  auto name=d.get_info<sycl::info::device::name>();
  json identity=json::object(); json device_id=nullptr;json optional_errors=json::array();
  auto optional=[&](const char* field,const auto& action){try{action();}catch(const sycl::exception& e){optional_errors.push_back({{"field",field},{"message",e.what()}});}};
#if defined(SYCL_EXT_INTEL_DEVICE_INFO)
  optional("pci_bus_id",[&]{if(d.has(sycl::aspect::ext_intel_pci_address))
    identity["pci_bus_id"]=d.get_info<sycl::ext::intel::info::device::pci_address>();});
#if SYCL_EXT_INTEL_DEVICE_INFO >= 2
  optional("uuid",[&]{if(d.has(sycl::aspect::ext_intel_device_info_uuid))
    identity["uuid"]=hex_bytes(d.get_info<sycl::ext::intel::info::device::uuid>());});
#endif
#if SYCL_EXT_INTEL_DEVICE_INFO >= 5
  optional("device_id",[&]{if(d.has(sycl::aspect::ext_intel_device_id))
    device_id=d.get_info<sycl::ext::intel::info::device::device_id>();});
#endif
#if SYCL_EXT_INTEL_DEVICE_INFO >= 8
  optional("luid",[&]{if(d.has(sycl::aspect::ext_intel_device_info_luid))
    identity["luid"]=hex_bytes(d.get_info<sycl::ext::intel::info::device::luid>());});
#endif
#endif
  auto id="intel:sycl:"+std::to_string(index);
  return {{"id",id},{"alias","intel:"+std::to_string(index)},{"backend_index",index},
          {"name",name},{"vendor_id",0x8086},{"device_id",device_id},
          {"architecture",architecture(name)},{"identity",identity},
          {"memory_bytes",d.get_info<sycl::info::device::global_mem_size>()},
          {"integrated",d.get_info<sycl::info::device::host_unified_memory>()},
          {"sycl_backend",backend_name(d)},
          {"driver",d.get_info<sycl::info::device::driver_version>()},
          {"platform",d.get_platform().get_info<sycl::info::platform::name>()},
          {"fp16",d.has(sycl::aspect::fp16)},
          {"device_usm",d.has(sycl::aspect::usm_device_allocations)},
          {"queue_profiling",d.has(sycl::aspect::queue_profiling)},
          {"single_fp_config",fp_config(d,false)},
          {"half_fp_config",d.has(sycl::aspect::fp16)?fp_config(d,true):json::array()},
          {"optional_query_errors",optional_errors},
          {"identity_scope",identity.empty()?"backend_qualified_only":"runtime_reported"}};
}

// Only LUID equality establishes a DXGI/SYCL identity. Names/indices never do.
std::optional<uint64_t> dxgi_available(const sycl::device& d) {
#if defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 8
  if(!d.has(sycl::aspect::ext_intel_device_info_luid))return {};
  auto bytes=d.get_info<sycl::ext::intel::info::device::luid>();
  LUID luid{};static_assert(sizeof(luid)==8);std::memcpy(&luid,bytes.data(),8);
  ComPtr<IDXGIFactory4> factory;
  if(FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))return {};
  ComPtr<IDXGIAdapter3> adapter;
  if(FAILED(factory->EnumAdapterByLuid(luid,IID_PPV_ARGS(&adapter))))return {};
  DXGI_QUERY_VIDEO_MEMORY_INFO mem{};
  if(FAILED(adapter->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&mem)))return {};
  return mem.Budget>mem.CurrentUsage?mem.Budget-mem.CurrentUsage:0;
#else
  (void)d;return {};
#endif
}
json memory_budget(const sycl::device& d,uint64_t reserve) {
  std::optional<uint64_t> free;
  std::string source;json query_errors=json::array();
#if defined(SYCL_EXT_INTEL_DEVICE_INFO) && SYCL_EXT_INTEL_DEVICE_INFO >= 4
  if(d.has(sycl::aspect::ext_intel_free_memory)) {
    try{free=d.get_info<sycl::ext::intel::info::device::free_memory>();source="sycl_ext_intel_free_memory";}
    catch(const sycl::exception& e){query_errors.push_back(e.what());}
  }
#endif
  try{if(auto v=dxgi_available(d)) {
      free=free?std::min(*free,*v):*v;
      source+=source.empty()?"dxgi_luid_budget":"+dxgi_luid_budget_min";
    }}catch(const sycl::exception& e){query_errors.push_back(e.what());}
  if(!free)throw gb::Error("dynamic_budget_unavailable","configuration_unsupported",
    "Neither SYCL free_memory nor LUID-matched DXGI current budget is available; total VRAM is not a dynamic budget. Query errors="+query_errors.dump());
  auto half=*free/2;
  if(half<=reserve)throw gb::Error("memory_budget","configuration_unsupported","Half the dynamic memory budget is below the library reserve");
  return {{"dynamic_available_bytes",*free},{"source",source},{"fraction",0.5},
          {"library_reserve_bytes",reserve},{"worker_allocation_limit_bytes",half-reserve},
          {"query_errors",query_errors},
          {"library_internal_workspace","opaque; reserve retained, not measured or user-controllable"}};
}

struct State {
  sycl::device device;
  sycl::queue queue;
  json description;
  bool lost=false;
  explicit State(sycl::device d,json desc): device(d),queue(d,[](sycl::exception_list es){
    for(auto e:es)std::rethrow_exception(e);
  },d.has(sycl::aspect::queue_profiling)
    ?sycl::property_list{sycl::property::queue::in_order{},sycl::property::queue::enable_profiling{}}
    :sycl::property_list{sycl::property::queue::in_order{}}),description(std::move(desc)) {}
};
gb::Error sycl_error(State& state,const sycl::exception& e) {
  auto m=lower(e.what());
  if(m.find("device lost")!=std::string::npos || m.find("device_lost")!=std::string::npos ||
     m.find("device reset")!=std::string::npos || m.find("device hung")!=std::string::npos ||
     m.find("device_not_available")!=std::string::npos || m.find("device removed")!=std::string::npos) {
    state.lost=true;return {"device_lost","driver_incompatible",e.what()};
  }
  if(e.code()==sycl::make_error_code(sycl::errc::memory_allocation))
    return {"out_of_memory","configuration_unsupported",e.what()};
  if(e.code()==sycl::make_error_code(sycl::errc::feature_not_supported))
    return {"api_unsupported","api_unsupported",e.what()};
  if(e.code()==sycl::make_error_code(sycl::errc::kernel_not_supported))
    return {"kernel_missing","kernel_missing",e.what()};
  if(e.code()==sycl::make_error_code(sycl::errc::build))
    return {"kernel_build_failed","kernel_missing",e.what()};
  return {"sycl_execution_error","probe_failed",e.what()};
}
void invalidate(json& result,const std::string& code,const std::string& status,const std::string& message) {
  result["validation_status"]="failed";result["result_status"]="failed";
  result["backend_status"]=status;result["reason_code"]=code;result["reason_message"]=message;
  result["median_tflops"]=nullptr;result["best_tflops"]=nullptr;
}
template<class T> struct DeviceArray {
  sycl::queue* queue;T* ptr=nullptr;size_t size;
  DeviceArray(sycl::queue& q,size_t n):queue(&q),size(n) {
    gb::checked_product({uint64_t(n),sizeof(T)});
    ptr=sycl::malloc_device<T>(n,q);
    if(!ptr)throw gb::Error("out_of_memory","configuration_unsupported","Device USM allocation returned nullptr");
  }
  DeviceArray(const DeviceArray&)=delete;
  ~DeviceArray(){if(ptr){try{queue->wait_and_throw();}catch(...){}try{sycl::free(ptr,*queue);}catch(...){}}}
};

// completion events are dependencies, never treated as full oneMKL intervals.
struct Interval {
  State& state;DeviceArray<uint64_t> marks;uint64_t serial=0;bool device_timing=true;
  explicit Interval(State& s):state(s),marks(s.queue,2),device_timing(s.device.has(sycl::aspect::queue_profiling)){}
  gb::BatchTime time(uint64_t reps,const std::function<sycl::event(uint64_t,const std::vector<sycl::event>&)>& submit) {
    auto& q=state.queue;
    q.wait_and_throw();auto wall0=gb::Clock::now();auto p=marks.ptr;auto seq=++serial;
    auto start=q.submit([&](sycl::handler& h){h.single_task([=](){p[0]=seq;});});
    sycl::event last=start;
    for(uint64_t r=0;r<reps;r++)last=submit(r,{last});
    auto stop=q.submit([&](sycl::handler& h){h.depends_on(last);h.single_task([=](){p[1]=seq;});});
    stop.wait_and_throw();q.wait_and_throw();auto wall=gb::millis(wall0,gb::Clock::now());
    double elapsed=wall;
    if(device_timing) {
      auto a=start.get_profiling_info<sycl::info::event_profiling::command_end>();
      auto b=stop.get_profiling_info<sycl::info::event_profiling::command_start>();
      if(b<=a)throw gb::Error("invalid_device_interval","probe_failed","Sentinel timestamps are non-positive or unordered");
      elapsed=double(b-a)/1e6;
      if(elapsed>wall*1.25+2.)throw gb::Error("profiling_wall_disagreement","probe_failed","Device interval exceeds synchronized wall time");
    }
    return {elapsed,wall};
  }
  void check_marks() {
    std::array<uint64_t,2> v{};state.queue.memcpy(v.data(),marks.ptr,sizeof(v)).wait_and_throw();
    if(v[0]!=serial||v[1]!=serial)throw gb::Error("sentinel_validation_failed","probe_failed","Real USM marker writes did not complete");
  }
  void select_timing(const std::function<sycl::event(uint64_t,const std::vector<sycl::event>&)>& submit,json& result) {
    try{time(1,submit);check_marks();}
    catch(const sycl::exception& e) {
      // Only profiling query failure permits a wall fallback. An execution
      // failure must still fail; a second synchronized run must pass.
      if(e.code()!=sycl::make_error_code(sycl::errc::profiling))throw;
      device_timing=false;result["device_timing_unavailable_reason"]=e.what();
      time(1,submit);check_marks();
    }
    catch(const gb::Error& e) {
      if(e.code!="invalid_device_interval"&&e.code!="profiling_wall_disagreement")throw;
      device_timing=false;result["device_timing_unavailable_reason"]=e.what();
      time(1,submit);check_marks();
    }
    if(!device_timing&&!result.contains("device_timing_unavailable_reason"))
      result["device_timing_unavailable_reason"]="SYCL queue_profiling aspect is absent";
    result["timing_source"]=device_timing?"sycl_profiling_sentinels":"host_qpc";
    result["timing_scope"]=device_timing?"device_elapsed_batch":"host_synchronized_batch";
    result["configuration"]["timing_validation"]={{"in_order",true},{"real_device_usm_writes",true},
      {"first_operation_depends_on_start",true},{"stop_depends_on_final_completion",true},
      {"interval","front_marker.command_end -> back_marker.command_start"},{"synchronized_wall_retained",true}};
  }
};

template<class T,int A,int U,int L> sycl::event alu_submit(sycl::queue& q,const T* input,
    const T* multiplier,const T* addend,T* output,size_t threads,size_t local,uint32_t iterations,
    const std::vector<sycl::event>& dependencies) {
  using V=std::conditional_t<L==2,sycl::vec<T,2>,T>;
  return q.submit([&](sycl::handler& h){
    h.depends_on(dependencies);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(threads),sycl::range<1>(local)),[=](sycl::nd_item<1> item){
      const size_t t=item.get_global_linear_id();V x[A];V m,b;
      if constexpr(L==2){m=V(multiplier[t*2],multiplier[t*2+1]);b=V(addend[t*2],addend[t*2+1]);}
      else {m=multiplier[t];b=addend[t];}
      #pragma unroll
      for(int a=0;a<A;a++) {
        if constexpr(L==2)x[a]=V(input[(t*A+a)*2],input[(t*A+a)*2+1]);
        else x[a]=input[t*A+a];
      }
      for(uint32_t i=0;i<iterations;i+=U) {
        #pragma unroll
        for(int u=0;u<U;u++) {
          #pragma unroll
          for(int a=0;a<A;a++)x[a]=sycl::fma(x[a],m,b);
        }
      }
      #pragma unroll
      for(int a=0;a<A;a++) {
        if constexpr(L==2){output[(t*A+a)*2]=x[a].x();output[(t*A+a)*2+1]=x[a].y();}
        else output[t*A+a]=x[a];
      }
    });
  });
}
template<class T> struct AluSet {
  DeviceArray<T> input,multiplier,addend,output;
  std::vector<T> hi,hm,hb;
  AluSet(sycl::queue& q,size_t max_threads,uint32_t seed):input(q,max_threads*32),
      multiplier(q,max_threads*2),addend(q,max_threads*2),output(q,max_threads*32),
      hi(input.size),hm(multiplier.size),hb(addend.size) {
    std::mt19937 rng(seed);
    for(auto& x:hi)x=T((float(int(rng()%2048)-1024))/4096.f);
    for(size_t i=0;i<hm.size();i++) {
      hm[i]=T(1.f-float(1+(rng()%2))/512.f);
      hb[i]=T(float(int(rng()%9)-4)/4096.f);
      if(float(hb[i])==0)hb[i]=T(1.f/4096.f);
    }
    q.memcpy(input.ptr,hi.data(),hi.size()*sizeof(T));
    q.memcpy(multiplier.ptr,hm.data(),hm.size()*sizeof(T));
    q.memcpy(addend.ptr,hb.data(),hb.size()*sizeof(T));q.wait_and_throw();
  }
};
struct AluConfig {size_t threads=0,local=0;int accum=0,unroll=0,lanes=0;uint32_t iterations=256;};
template<class T> sycl::event dispatch_alu(sycl::queue& q,AluSet<T>& b,const AluConfig& c,
    const std::vector<sycl::event>& deps) {
  #define ALU(A,U,L) return alu_submit<T,A,U,L>(q,b.input.ptr,b.multiplier.ptr,b.addend.ptr,b.output.ptr,c.threads,c.local,c.iterations,deps)
  #define PICK_L(A,U) if(c.lanes==2){ALU(A,U,2);}else{ALU(A,U,1);}
  #define PICK_U(A) if(c.unroll==8){PICK_L(A,8);}else{PICK_L(A,4);}
  if(c.accum==16){PICK_U(16);}else if(c.accum==8){PICK_U(8);}else{PICK_U(4);}
  #undef PICK_U
  #undef PICK_L
  #undef ALU
}
template<class T> double scalar_reference(T initial,T m,T b,uint32_t iterations) {
  T x=initial;
  for(uint32_t i=0;i<iterations;i++) {
    if constexpr(std::is_same_v<T,float>)x=std::fma(x,m,b);
    else x=T(float(std::fma(double(float(x)),double(float(m)),double(float(b)))));
  }
  return double(float(x));
}
template<class T> json validate_alu(State& state,AluSet<T>& b,const AluConfig& c,bool all) {
  auto count=gb::checked_product({uint64_t(c.threads),uint64_t(c.accum),uint64_t(c.lanes)});
  std::vector<T> output(static_cast<size_t>(count));
  state.queue.memcpy(output.data(),b.output.ptr,output.size()*sizeof(T)).wait_and_throw();
  double checksum=0,max_abs=0;size_t finite=0,checked=0;
  for(auto x:output){if(!std::isfinite(float(x)))throw gb::Error("validation_failed","probe_failed","ALU produced non-finite output");checksum+=float(x);finite++;}
  auto n=all?count:std::min<uint64_t>(64,count);
  for(uint64_t s=0;s<n;s++) {
    auto i=all?s:(s*(count-1)/std::max<uint64_t>(1,n-1));
    auto thread=i/uint64_t(c.accum*c.lanes),lane=i%uint64_t(c.lanes);
    double ref=scalar_reference(b.hi[size_t(i)],b.hm[size_t(thread*c.lanes+lane)],b.hb[size_t(thread*c.lanes+lane)],c.iterations);
    double err=std::abs(double(float(output[size_t(i)]))-ref);
    double unit=std::is_same_v<T,float>?std::ldexp(1.,-23):std::ldexp(1.,-10);
    if(err>1e-7+2*unit*std::abs(ref))throw gb::Error("validation_failed","probe_failed",
      "ALU reference mismatch at index="+std::to_string(i)+" actual="+std::to_string(float(output[size_t(i)]))+" expected="+std::to_string(ref));
    max_abs=std::max(max_abs,err);checked++;
  }
  return {{"finite_elements",finite},{"reference_elements",checked},{"checksum",checksum},{"max_absolute_error",max_abs}};
}

template<class T> json run_alu(State& state,json& result,const json& request,std::atomic_bool& cancel,const gb::Progress& progress) {
  progress({{"phase","preparing_alu"}});
  auto budget=memory_budget(state.device,0);auto groups=request.value("cache","reuse")=="rotate"?3u:1u;
  result["configuration"]["memory_budget"]=budget;result["configuration"]["cache_buffer_groups"]=groups;
  auto limit=budget["worker_allocation_limit_bytes"].get<uint64_t>();
  auto max_group=state.device.get_info<sycl::info::device::max_work_group_size>();
  auto compute=state.device.get_info<sycl::info::device::max_compute_units>();
  size_t max_threads=std::min<uint64_t>(262144,(limit>16?limit-16:0)/(groups*68*sizeof(T)));
  max_threads=std::min<size_t>(max_threads,std::max<size_t>(4096,size_t(compute)*256*8));
  max_threads=max_threads/256*256;
  if(max_threads<256)throw gb::Error("memory_budget","configuration_unsupported","Insufficient half-budget for ALU workgroups");
  std::vector<std::unique_ptr<AluSet<T>>> sets;
  // Allocation failure reduces thread geometry, never the counted iteration
  // definition. The selected geometry and every fallback are recorded.
  json fallbacks=json::array();
  for(;;) {
    try{for(unsigned g=0;g<groups;g++)sets.push_back(std::make_unique<AluSet<T>>(state.queue,max_threads,request.value("seed",20261020u)+g));break;}
    catch(const sycl::exception& e){sets.clear();if(e.code()!=sycl::make_error_code(sycl::errc::memory_allocation))throw;}
    catch(const gb::Error& e){sets.clear();if(e.code!="out_of_memory")throw;}
    fallbacks.push_back({{"threads",max_threads},{"reason","allocation_failed"}});
    max_threads=max_threads/2/256*256;
    if(max_threads<256)throw gb::Error("out_of_memory","configuration_unsupported","ALU allocation failed at smallest geometry");
  }
  Interval interval(state);AluConfig best;double score=-1;json trials=json::array();
  for(size_t local:{64u,128u,256u})for(int accum:{4,8,16})for(int unroll:{4,8})for(int lanes:{1,2}) {
    if(cancel)break;
    if(local>max_group || (std::is_same_v<T,float>&&lanes==2))continue;
    AluConfig c{max_threads/local*local,local,accum,unroll,lanes,256};
    auto submit=[&](uint64_t,const std::vector<sycl::event>& deps){return dispatch_alu(state.queue,*sets[0],c,deps);};
    if(best.threads==0)interval.select_timing(submit,result);
    auto t=interval.time(1,submit);
    double ops=double(c.threads)*accum*lanes*c.iterations*2;
    trials.push_back({{"workgroup_size",local},{"independent_accumulators",accum},
      {"unroll",unroll},{"lanes_per_instruction",lanes},{"elapsed_ms",t.device_ms},{"phase","tuning"}});
    progress({{"phase","tuning"},{"completed_configurations",trials.size()}});
    if(t.device_ms>0&&t.device_ms<=100&&ops/t.device_ms>score){score=ops/t.device_ms;best=c;}
  }
  if(cancel){result["result_status"]="cancelled";result["reason_code"]="user_cancelled";return result;}
  if(!best.threads)throw gb::Error("kernel_missing","kernel_missing","No legal workgroup configuration");
  result["configuration"].update({{"threads",best.threads},{"workgroup_size",best.local},
    {"independent_accumulators",best.accum},{"unroll",best.unroll},{"lanes_per_instruction",best.lanes},
    {"iterations",best.iterations},{"tuning_trials",trials},{"allocation_fallbacks",fallbacks}});
  // Check every element on a short exact geometry, then full iterations on
  // sampled lanes with an all-element finite/readback check.
  auto short_c=best;short_c.threads=best.local;short_c.iterations=8;
  dispatch_alu(state.queue,*sets[0],short_c,{}).wait_and_throw();
  auto short_validation=validate_alu(state,*sets[0],short_c,true);
  double calibrated=0;
  for(;;) {
    auto t=interval.time(1,[&](uint64_t,const std::vector<sycl::event>& deps){return dispatch_alu(state.queue,*sets[0],best,deps);});
    calibrated=t.device_ms;
    if(calibrated>=1. || best.iterations>=65536)break;
    best.iterations*=2;
  }
  if(calibrated>20) {
    best.iterations=std::max(8u,(uint32_t(double(best.iterations)*5/calibrated)/8)*8);
  } else if(calibrated<20&&calibrated>0) {
    best.iterations=std::clamp((uint32_t(double(best.iterations)*5/calibrated)/8)*8,8u,65536u);
  }
  result["configuration"]["iterations"]=best.iterations;
  json validation=json::array();
  for(auto& s:sets){dispatch_alu(state.queue,*s,best,{}).wait_and_throw();validation.push_back(validate_alu(state,*s,best,false));}
  result["validation_status"]="passed";result["reference_kind"]="CPU fused per-step rounded recurrence";
  result["sampled_elements"]=64*groups;result["error_metrics"]={{"short_complete",short_validation},{"full_length",validation}};
  result["tolerance_policy"]={{"id","alu-fma-round-v1"},{"absolute_floor",1e-7},{"relative_units",2},
    {"unit",std::is_same_v<T,float>?std::ldexp(1.,-23):std::ldexp(1.,-10)}};
  result["configuration"].update({{"threads",best.threads},{"workgroup_size",best.local},
    {"independent_accumulators",best.accum},{"unroll",best.unroll},{"iterations",best.iterations},
    {"lanes_per_instruction",best.lanes},{"kernel_variant",best.lanes==2?"sycl_half2_fma":"sycl_scalar_fma"},
    {"memory_budget",budget},{"cache_buffer_groups",groups},{"allocation_fallbacks",fallbacks},
    {"allocated_bytes",groups*max_threads*68*sizeof(T)+16},{"workspace_bytes",0},
    {"tuning_trials",trials},{"searched_configurations",trials.size()}});
  result["native_evidence"]=json::array({{{"level","documented_api_path"},{"api","sycl::fma with native FMA fp_config and non-software GPU"},
    {"source","https://raw.githubusercontent.com/intel/llvm/sycl/sycl/include/sycl/info/device.hpp"},
    {"operand_type",std::is_same_v<T,float>?"float":"sycl::half"},{"isa_checked",false},{"profiler_checked",false},
    {"certification","instruction inspection and spill/register audit still required"}}});
  uint64_t batch_index=0;
  auto batch=[&](uint64_t reps){auto group=(batch_index++)%groups;
    try{return interval.time(reps,[&](uint64_t r,const std::vector<sycl::event>& deps){return dispatch_alu(state.queue,*sets[(group+r)%groups],best,deps);});}
    catch(const sycl::exception& e){throw sycl_error(state,e);}
    catch(const gb::Error&){throw;}
    catch(const std::exception& e){throw gb::Error("alu_execution_error","probe_failed",e.what());}};
  auto ops=gb::checked_product({uint64_t(best.threads),uint64_t(best.accum),uint64_t(best.lanes),best.iterations,2});
  result=gb::measure(std::move(result),request,cancel,ops,batch,progress);
  try {
    interval.check_marks();json after=json::array();
    for(auto& s:sets)after.push_back(validate_alu(state,*s,best,false));
    result["error_metrics"]["post_measurement"]=after;
  }catch(const sycl::exception& e){auto mapped=sycl_error(state,e);invalidate(result,mapped.code,mapped.status,mapped.what());}
  catch(const std::exception& e){invalidate(result,"post_validation_failed","probe_failed",e.what());}
  return result;
}

template<class T> struct GemmSet {
  int64_t m,n,k;DeviceArray<T> a,b;DeviceArray<float> c;std::vector<T> ha,hb;double quantization_max=0;
  GemmSet(sycl::queue& q,int64_t rows,int64_t cols,int64_t inner,uint32_t seed,bool exact=false):m(rows),n(cols),k(inner),
    a(q,size_t(gb::checked_product({uint64_t(m),uint64_t(k)}))),
    b(q,size_t(gb::checked_product({uint64_t(k),uint64_t(n)}))),
    c(q,size_t(gb::checked_product({uint64_t(m),uint64_t(n)}))),ha(a.size),hb(b.size) {
    std::mt19937 rng(seed);
    auto initialize=[&](auto& values){for(auto& x:values) {
      // FP32 random inputs retain 23 significant fraction bits, so the strict
      // numerical probe can actually distinguish truncated TF32/BF16 inputs.
      float original=exact?float(int(rng()%7)-3):float(int(rng()%16777215)-8388607)/8388608.f;
      x=T(original);quantization_max=std::max(quantization_max,std::abs(double(float(x))-original));
    }};
    initialize(ha);initialize(hb);
    q.memcpy(a.ptr,ha.data(),a.size*sizeof(T));q.memcpy(b.ptr,hb.data(),b.size*sizeof(T));q.wait_and_throw();
  }
  sycl::event submit(sycl::queue& q,const std::vector<sycl::event>& deps) {
    // 2026 USM mixed signature: Ta/Tb inferred as half, Tc/Ts as float.
    // A,B,C are row major, N/N, lda=K, ldb=N, ldc=N; C = A*B.
    return oneapi::mkl::blas::row_major::gemm(q,oneapi::mkl::transpose::nontrans,
      oneapi::mkl::transpose::nontrans,m,n,k,1.0f,a.ptr,k,b.ptr,n,0.0f,c.ptr,n,
      oneapi::mkl::blas::compute_mode::standard,deps);
  }
  json validate(sycl::queue& q,bool complete,bool exact=false) {
    std::vector<float> output(c.size);q.memcpy(output.data(),c.ptr,c.size*sizeof(float)).wait_and_throw();
    auto policy=gb::tolerance_policy(std::is_same_v<T,float>?"fp32":"fp16",uint64_t(k));
    double max_abs=0,max_normalized=0,checksum=0;uint64_t checked=0;
    for(auto v:output){if(!std::isfinite(v))throw gb::Error("validation_failed","probe_failed","GEMM produced non-finite output");checksum+=v;}
    auto count=complete?uint64_t(c.size):std::min<uint64_t>(97,c.size);
    json checked_values=json::array();
    for(uint64_t sample=0;sample<count;sample++) {
      uint64_t pos=complete?sample:sample*(c.size-1)/std::max<uint64_t>(1,count-1);
      auto row=pos/uint64_t(n),col=pos%uint64_t(n);long double ref=0,sum_abs=0;
      for(int64_t z=0;z<k;z++) {auto product=static_cast<long double>(float(ha[size_t(row*k+z)]))*float(hb[size_t(z*n+col)]);ref+=product;sum_abs+=std::abs(product);}
      double actual=output[size_t(pos)],expected=double(ref),error=std::abs(actual-expected);
      if((exact&&actual!=expected)||!gb::valid_dot(actual,expected,double(sum_abs),policy))
        throw gb::Error("validation_failed","probe_failed","GEMM disagrees with quantized-input CPU reference at row="+std::to_string(row)+" col="+std::to_string(col)+
          " actual="+std::to_string(actual)+" expected="+std::to_string(expected));
      max_abs=std::max(max_abs,error);max_normalized=std::max(max_normalized,error/std::max(1e-12,double(sum_abs)));checked++;
      if(!complete)checked_values.push_back({{"row",row},{"col",col},{"actual",actual},{"reference",expected},{"sum_absolute_products",double(sum_abs)}});
    }
    return {{"reference_elements",checked},{"finite_elements",output.size()},{"max_absolute_error",max_abs},
      {"max_normalized_error",max_normalized},{"checksum",checksum},{"tolerance_policy",policy},
      {"quantization_error",{{"input_max_absolute_error",quantization_max}}},
      {"quantization_note","input quantization is reported separately; kernel reference uses actual stored FP16/FP32 values"},
      {"sampled_output",checked_values}};
  }
};
template<class T> json probe_execution(State& state,const std::string& suite) {
  json result={{"configuration",json::object()}};Interval interval(state);json validation;
  if(suite=="gemm") {
    GemmSet<T> values(state.queue,17,19,13,20261020);
    auto submit=[&](uint64_t,const std::vector<sycl::event>& deps){return values.submit(state.queue,deps);};
    interval.select_timing(submit,result);validation=values.validate(state.queue,true);
  } else {
    auto local=std::min<size_t>(64,state.device.get_info<sycl::info::device::max_work_group_size>());
    AluSet<T> values(state.queue,local,20261020);AluConfig config{local,local,4,4,1,8};
    auto submit=[&](uint64_t,const std::vector<sycl::event>& deps){return dispatch_alu(state.queue,values,config,deps);};
    interval.select_timing(submit,result);validation=validate_alu(state,values,config,true);
  }
  return {{"validation_status","passed"},{"validation",validation},{"timing_source",result["timing_source"]},
    {"timing_scope",result["timing_scope"]},{"timing_validation",result["configuration"]["timing_validation"]},
    {"native_evidence_level","documented_api_path"},{"isa_checked",false},{"profiler_checked",false}};
}
template<class T> json run_gemm(State& state,json& result,const json& request,std::atomic_bool& cancel,const gb::Progress& progress) {
  progress({{"phase","preparing_gemm"}});
  int64_t m=request.value("m",1024ll),n=request.value("n",1024ll),k=request.value("k",1024ll);
  if(m<=0||n<=0||k<=0||m>INT32_MAX||n>INT32_MAX||k>INT32_MAX)
    throw gb::Error("invalid_shape","configuration_unsupported","Positive LP64 matrix dimensions are required");
  auto ops=gb::checked_product({2,uint64_t(m),uint64_t(n),uint64_t(k)});
  auto seed=request.value("seed",20261020u);auto groups=request.value("cache","reuse")=="rotate"?3u:1u;
  auto reserve=std::max<uint64_t>(64ull*1024*1024,gb::checked_product({request.value("workspace_mib",256ull),1024,1024}));
  auto budget=memory_budget(state.device,reserve);
  result["configuration"]["memory_budget"]=budget;result["configuration"]["cache_buffer_groups"]=groups;
  auto ab=add_bytes(gb::checked_product({uint64_t(m),uint64_t(k),sizeof(T)}),gb::checked_product({uint64_t(k),uint64_t(n),sizeof(T)}));
  auto cb=gb::checked_product({uint64_t(m),uint64_t(n),sizeof(float)});
  auto bytes=gb::checked_product({add_bytes(ab,cb),groups});
  if(add_bytes(bytes,16)>budget["worker_allocation_limit_bytes"].get<uint64_t>())
    throw gb::Error("memory_budget","configuration_unsupported","Explicit shape and cache groups exceed half the current memory budget; dimensions were not changed");
  // Exact integers plus random non-square, odd boundary matrices catch transpose
  // or stride errors before any large user-selected shape can be sampled.
  json preflight=json::array();
  for(auto shape: {std::array<int64_t,3>{3,5,7},std::array<int64_t,3>{17,19,13},std::array<int64_t,3>{32,32,32}}) {
    GemmSet<T> test(state.queue,shape[0],shape[1],shape[2],seed,true);
    test.submit(state.queue,{}).wait_and_throw();preflight.push_back({{"shape",shape},{"validation",test.validate(state.queue,true,true)}});
  }
  {GemmSet<T> random(state.queue,23,29,31,seed+1);random.submit(state.queue,{}).wait_and_throw();
    preflight.push_back({{"shape",{23,29,31}},{"validation",random.validate(state.queue,true)}});}
  if(cancel){result["result_status"]="cancelled";result["reason_code"]="user_cancelled";return result;}
  // Grow each logical dimension from a small legal shape. Reject before the
  // next launch if measured throughput projects a dispatch above 100 ms.
  // This is a conservative safety estimate, never a reported benchmark score.
  json safety=json::array();
  {
    Interval safety_timer(state);double previous_ms=0;uint64_t previous_ops=0;
    for(int64_t bound=64;;bound*=2) {
      auto cm=std::min(m,bound),cn=std::min(n,bound),ck=std::min(k,bound);
      auto cops=gb::checked_product({2,uint64_t(cm),uint64_t(cn),uint64_t(ck)});
      if(previous_ops&&previous_ms*double(cops)/double(previous_ops)>100.) {
        result["configuration"]["safety_calibration"]=safety;
        return gb::unavailable(result,result["hardware_support"],"configuration_unsupported","dispatch_budget",
          "Measured smaller-shape throughput projects the next GEMM dispatch above 100 ms; explicit dimensions were not changed");
      }
      GemmSet<T> small(state.queue,cm,cn,ck,seed);
      auto call=[&](uint64_t,const std::vector<sycl::event>& deps){return small.submit(state.queue,deps);};
      if(!previous_ops)safety_timer.select_timing(call,result);
      small.submit(state.queue,{}).wait_and_throw();
      auto t=safety_timer.time(1,call);
      safety.push_back({{"shape",{cm,cn,ck}},{"elapsed_ms",t.device_ms},{"wall_ms",t.wall_ms},
        {"timing_scope",result["timing_scope"]},{"purpose","dispatch_safety; excluded from formal samples"}});
      progress({{"phase","dispatch_safety"},{"shape",{cm,cn,ck}},{"elapsed_ms",t.device_ms}});
      if(t.device_ms>100.) {
        result["configuration"]["safety_calibration"]=safety;
        return gb::unavailable(result,result["hardware_support"],"configuration_unsupported","dispatch_budget",
          "A single calibration GEMM exceeded 100 ms; no formal samples are submitted");
      }
      if(cancel){result["result_status"]="cancelled";result["reason_code"]="user_cancelled";return result;}
      previous_ms=t.device_ms;previous_ops=cops;
      if(cm==m&&cn==n&&ck==k)break;
    }
  }
  std::vector<std::unique_ptr<GemmSet<T>>> sets;
  // Explicit shapes must never silently shrink, including on allocation failure.
  for(unsigned g=0;g<groups;g++)sets.push_back(std::make_unique<GemmSet<T>>(state.queue,m,n,k,seed+g));
  Interval interval(state);auto submit=[&](uint64_t r,const std::vector<sycl::event>& deps){return sets[size_t(r%groups)]->submit(state.queue,deps);};
  interval.select_timing(submit,result);
  json initial=json::array();
  for(auto& set:sets){set->submit(state.queue,{}).wait_and_throw();initial.push_back(set->validate(state.queue,m*n<=4096));}
  result["validation_status"]="passed";result["reference_kind"]="CPU long double dot on actual quantized inputs";
  result["sampled_elements"]=(m*n<=4096?uint64_t(m)*n:std::min<uint64_t>(97,uint64_t(m)*n))*groups;
  result["error_metrics"]={{"preflight",preflight},{"performance_shape",initial}};
  result["tolerance_policy"]=gb::tolerance_policy(std::is_same_v<T,float>?"fp32":"fp16",uint64_t(k));
  result["configuration"].update({{"M",m},{"N",n},{"K",k},{"logical_shape",{m,n,k}},
    {"physical_shape",{m,n,k}},{"layout","row_major"},{"transa","N"},{"transb","N"},
    {"lda",k},{"ldb",n},{"ldc",n},{"alpha",1},{"beta",0},
    {"compute_mode","oneapi::mkl::blas::compute_mode::standard"},
    {"usm_allocation","device"},{"library","oneMKL SYCL BLAS"},
    {"algorithm_id",nullptr},{"algorithm_selection","oneMKL opaque dispatch"},
    {"workspace_bytes",nullptr},{"workspace_user_control",false},
    {"memory_budget",budget},{"allocated_bytes",bytes+16},{"cache_buffer_groups",groups},
    {"safety_calibration",safety},
    {"searched_configurations",1},{"xmx_verified",false}});
  result["native_evidence"]=json::array({{{"level","documented_api_path"},{"source",gemm_source},
    {"compute_mode_source",mode_source},{"api","oneapi::mkl::blas::row_major::gemm USM"},
    {"input_type",std::is_same_v<T,float>?"float":"sycl::half"},{"accumulation","fp32"},{"output_type","float"},
    {"isa_checked",false},{"profiler_checked",false},{"xmx_claimed",false}}});
  uint64_t base=0;
  auto batch=[&](uint64_t reps){auto first=base;base+=reps;
    try{return interval.time(reps,[&](uint64_t r,const std::vector<sycl::event>& deps){return sets[size_t((first+r)%groups)]->submit(state.queue,deps);});}
    catch(const sycl::exception& e){throw sycl_error(state,e);}
    catch(const oneapi::mkl::unimplemented& e){throw gb::Error("kernel_missing","kernel_missing",e.what());}
    catch(const gb::Error&){throw;}
    catch(const std::exception& e){throw gb::Error("onemkl_execution_error","probe_failed",e.what());}};
  result=gb::measure(std::move(result),request,cancel,ops,batch,progress);
  try{interval.check_marks();json after=json::array();for(auto& set:sets)after.push_back(set->validate(state.queue,m*n<=4096));result["error_metrics"]["post_measurement"]=after;}
  catch(const sycl::exception& e){auto mapped=sycl_error(state,e);invalidate(result,mapped.code,mapped.status,mapped.what());}
  catch(const std::exception& e){invalidate(result,"post_validation_failed","probe_failed",e.what());}
  return result;
}

class Backend final:public gb::Backend {
  std::vector<sycl::device> devices;
  std::vector<json> descriptions;
  std::vector<std::unique_ptr<State>> states;
  bool discovered=false;
  void discover() {
    if(discovered)return;
    auto platforms=sycl::platform::get_platforms();std::set<std::string> seen_uuid;
    // Prefer the Level Zero path when an actual UUID proves two API views are
    // the same GPU. Without proof preserve both backend-qualified identities.
    std::stable_sort(platforms.begin(),platforms.end(),[](const auto& a,const auto& b){
      return (a.get_backend()==sycl::backend::ext_oneapi_level_zero)>(b.get_backend()==sycl::backend::ext_oneapi_level_zero);
    });
    for(auto& p:platforms) {
      std::vector<sycl::device> gpu;
      try{gpu=p.get_devices(sycl::info::device_type::gpu);}catch(const sycl::exception& e){std::cerr<<"SYCL platform enumeration: "<<e.what()<<'\n';continue;}
      for(auto& d:gpu) {
        try {
          if(!d.is_gpu()||d.has(sycl::aspect::emulated)||d.get_info<sycl::info::device::vendor_id>()!=0x8086)continue;
          auto n=d.get_info<sycl::info::device::name>();if(excluded_product(n))continue;
          auto desc=describe(d,devices.size());auto identity=desc["identity"];
          if(identity.contains("uuid")) {auto key=identity["uuid"].get<std::string>();if(!seen_uuid.insert(key).second)continue;}
          devices.push_back(d);descriptions.push_back(desc);states.push_back(nullptr);
        }catch(const sycl::exception& e){std::cerr<<"SYCL GPU discovery: "<<e.what()<<'\n';}
      }
    }
    discovered=true;
  }
  size_t select(const json& request) {
    discover();if(devices.empty())throw gb::Error("gpu_not_found","configuration_unsupported","No Intel consumer GPU is exposed by SYCL; CPU fallback is disabled");
    const auto& given=request.at("device");auto id=given.at("id").get<std::string>();
    for(size_t i=0;i<descriptions.size();i++)if(descriptions[i]["id"]==id)return i;
    throw gb::Error("gpu_not_found","configuration_unsupported","Requested backend-qualified Intel device is absent");
  }
  State& state(size_t index) {
    if(!states[index]) {
      if(!devices[index].has(sycl::aspect::usm_device_allocations))throw gb::Error("device_usm_missing","api_unsupported","Device USM is required; shared/CPU allocations are not substituted");
      auto v=runtime_version();
      if(v["major"]!=__INTEL_MKL__||v["minor"]!=__INTEL_MKL_MINOR__||v["update"]!=__INTEL_MKL_UPDATE__)
        throw gb::Error("dll_version_mismatch","driver_incompatible","Loaded oneMKL core version does not match the compile-time oneMKL headers");
      states[index]=std::make_unique<State>(devices[index],descriptions[index]);
    }
    if(states[index]->lost)throw gb::Error("device_lost","driver_incompatible","Device previously lost; no further tests are submitted");
    return *states[index];
  }
 public:
  std::string name()const override{return "intel";}
  json enumerate()override{discover();return descriptions;}
  json probe(const json& request)override {
    auto index=select(request);auto& d=devices[index];auto desc=descriptions[index];auto arch=desc["architecture"].get<std::string>();
    json caps=json::array();
    for(auto suite:{"alu","gemm"})for(auto dtype:{"fp32","fp16","fp8"}) {
      bool half=std::string(dtype)=="fp16";
      std::string hw="unknown",status="ready",reason="runtime execution and numerical validation required";
      if(std::string(dtype)=="fp8") {
        hw=(arch=="arc_a"||arch=="arc_b")?"unsupported":"unknown";status="api_unsupported";
        reason=hw=="unsupported"?"Arc A/B lack a native FP8 execution path; conversions are not implemented":"Unrecognized architecture; this worker has no legal native FP8 oneMKL API path";
      } else if(half&&!d.has(sycl::aspect::fp16)) {status="api_unsupported";reason="SYCL aspect::fp16 is absent; hardware capability is not inferred";}
      else if(std::string(suite)=="alu"&&!has_fma(d,half)){status="api_unsupported";reason="Native FMA fp_config is absent or soft_float is reported";}
      else if(arch=="arc_a"||arch=="arc_b")hw="supported";
      if(!d.has(sycl::aspect::usm_device_allocations)){status="api_unsupported";reason="Device USM is required";}
      json execution=nullptr;
      if(status=="ready") {
        try {
          auto& s=state(index);
          execution=half?probe_execution<sycl::half>(s,suite):probe_execution<float>(s,suite);
          hw="supported";reason="Short GPU execution and numerical reference validation passed; ISA certification remains pending";
        }catch(const sycl::exception& e){auto mapped=states[index]?sycl_error(*states[index],e):gb::Error("sycl_initialization_failed","driver_incompatible",e.what());status=mapped.status;reason=mapped.what();}
        catch(const oneapi::mkl::unimplemented& e){status="kernel_missing";reason=e.what();}
        catch(const gb::Error& e){status=e.status;reason=e.what();}
      }
      caps.push_back({{"suite",suite},{"dtype",dtype},{"hardware_support",hw},{"backend_status",status},
        {"implementation",execution.is_null()?"unknown":"native"},{"reason",reason},{"execution_probe",execution}});
    }
    return {{"device",desc},{"capabilities",caps},{"runtime",{{"oneMKL",__INTEL_MKL__},
      {"oneMKL_minor",__INTEL_MKL_MINOR__},{"oneMKL_update",__INTEL_MKL_UPDATE__},
      {"compiler",__INTEL_LLVM_COMPILER},{"SYCL_LANGUAGE_VERSION",SYCL_LANGUAGE_VERSION},
      {"runtime_core",runtime_version()},{"loaded_modules",gb::loaded_modules()}}},{"cpu_fallback",false},
      {"matrix_micro",{{"backend_status","api_unsupported"},{"reason","Intel matrix_micro is outside the first release"}}}};
  }
  json run(const json& request,std::atomic_bool& cancel,const gb::Progress& progress)override {
    auto index=select(request);auto result=gb::base_result(descriptions[index],request,name());
    auto dtype=request.value("dtype","fp32"),suite=request.value("suite","alu");auto arch=descriptions[index]["architecture"].get<std::string>();
    if(dtype=="fp8")return gb::unavailable(result,(arch=="arc_a"||arch=="arc_b")?"unsupported":"unknown","api_unsupported","native_fp8_unavailable",
      arch=="arc_a"||arch=="arc_b"?"Arc A/B native FP8 is unsupported; no conversion benchmark is run":"Unknown architecture; no legal native FP8 API path in this worker");
    if(dtype!="fp32"&&dtype!="fp16")return gb::unavailable(result,"unknown","configuration_unsupported","dtype_out_of_scope","This release accepts FP32, FP16, and FP8 capability requests only");
    if(suite!="alu"&&suite!="gemm")return gb::unavailable(result,"unknown","api_unsupported","suite_unavailable","Intel first release implements alu and gemm");
    auto& d=devices[index];bool half=dtype=="fp16";
    if(half&&!d.has(sycl::aspect::fp16))return gb::unavailable(result,"unknown","api_unsupported","fp16_aspect_missing","SYCL aspect::fp16 is absent");
    if(suite=="alu"&&!has_fma(d,half))return gb::unavailable(result,"unknown","api_unsupported","native_fma_missing","Native FMA fp_config is absent or reports soft_float");
    result["hardware_support"]=(arch=="arc_a"||arch=="arc_b")?"supported":"unknown";
    result["implementation"]="native";
    result["build"]["oneMKL"]={{"major",__INTEL_MKL__},{"minor",__INTEL_MKL_MINOR__},{"update",__INTEL_MKL_UPDATE__}};
    result["build"]["intel_llvm_compiler"]=__INTEL_LLVM_COMPILER;
    result["build"]["sycl_language"]=SYCL_LANGUAGE_VERSION;
    result["build"]["oneMKL_runtime_core"]=runtime_version();
    result["configuration"]["compile_options"]="-fsycl -fno-fast-math -ffp-contract=on -fsycl-device-code-split=per_kernel";
    result["configuration"]["effective_environment"]={{"MKL_VERBOSE",0},{"MKL_BLAS_COMPUTE_MODE","STANDARD"},{"ONEMKL_VERBOSE",0}};
    try {
      auto& s=state(index);
      if(suite=="alu")result=half?run_alu<sycl::half>(s,result,request,cancel,progress):run_alu<float>(s,result,request,cancel,progress);
      else result=half?run_gemm<sycl::half>(s,result,request,cancel,progress):run_gemm<float>(s,result,request,cancel,progress);
      if(result["result_status"]=="ok")result["hardware_support"]="supported";
      result["build"]["loaded_modules"]=gb::loaded_modules();
      return result;
    }catch(const sycl::exception& e) {
      auto mapped=states[index]?sycl_error(*states[index],e):gb::Error("sycl_initialization_failed","driver_incompatible",e.what());
      result=gb::unavailable(result,result["hardware_support"],mapped.status,mapped.code,mapped.what());
      if(mapped.status=="probe_failed"||mapped.code=="device_lost")result["result_status"]="failed";
    }catch(const oneapi::mkl::unimplemented& e){result=gb::unavailable(result,result["hardware_support"],"kernel_missing","kernel_missing",e.what());}
    catch(const std::bad_alloc& e){result=gb::unavailable(result,result["hardware_support"],"configuration_unsupported","host_out_of_memory",e.what());}
    catch(const gb::Error& e) {
      result=gb::unavailable(result,result["hardware_support"],e.status,e.code,e.what());
      if(e.status=="probe_failed"){result["result_status"]="failed";if(e.code=="validation_failed")result["validation_status"]="failed";}
    }
    return result;
  }
};
} // namespace intel

int main() {
  // The launcher also sets these before process initialization. Override only
  // this worker's environment; diagnostics never enter JSONL stdout.
  SetEnvironmentVariableA("MKL_VERBOSE","0");SetEnvironmentVariableA("ONEMKL_VERBOSE","0");
  SetEnvironmentVariableA("MKL_BLAS_COMPUTE_MODE","STANDARD");
  intel::Backend backend;return gb::worker_main(backend);
}
