#pragma once
#include <nlohmann/json.hpp>
#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
#include <psapi.h>
namespace gb {
using json = nlohmann::json;
using Clock = std::chrono::steady_clock;
inline int64_t unix_ms(){return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
inline double millis(Clock::time_point a, Clock::time_point b) {return std::chrono::duration<double,std::milli>(b-a).count();}
struct Error: std::runtime_error {
  std::string code, status;
  Error(std::string c,std::string s,std::string m):std::runtime_error(m),code(std::move(c)),status(std::move(s)){}
};
inline uint64_t checked_product(std::initializer_list<uint64_t> values) {
  uint64_t n=1; for(auto v:values){if(v && n>UINT64_MAX/v) throw Error("count_overflow","configuration_unsupported","FLOP or allocation count overflows uint64"); n*=v;} return n;
}
inline double median(std::vector<double> v) {if(v.empty())throw std::invalid_argument("empty samples");std::sort(v.begin(),v.end());auto n=v.size();return n%2?v[n/2]:(v[n/2-1]+v[n/2])/2;}
inline double half_decode(uint16_t x){auto s=(x&0x8000)?-1.:1.;auto e=(x>>10)&31;auto m=x&1023;if(e==31)return m?NAN:s*INFINITY;return s*(e?std::ldexp(1.+m/1024.,int(e)-15):std::ldexp(double(m),-24));}
inline double e4m3_decode(uint8_t x) {double s=x&128?-1.:1.;int e=(x>>3)&15,m=x&7;if(e==15&&m==7)return NAN;return s*(e?std::ldexp(1.+m/8.,e-7):std::ldexp(double(m),-9));}
inline uint8_t e4m3_encode(double x) {if(!std::isfinite(x))throw std::invalid_argument("finite E4M3 input required");uint8_t best=0;double err=INFINITY;for(unsigned i=0;i<256;i++){double v=e4m3_decode(uint8_t(i)),d=std::abs(v-x);if(d<err||(d==err&&!(i&1))){err=d;best=uint8_t(i);}}return best;}
inline double reference_dot(const std::vector<double>& a,const std::vector<double>& b,size_t m,size_t n,size_t k,size_t row,size_t col){(void)m;long double s=0;for(size_t z=0;z<k;z++)s+=(long double)a.at(row*k+z)*b.at(z*n+col);return double(s);}
inline json tolerance_policy(const std::string& dtype,uint64_t k){double u=std::ldexp(1.,-24),ku=double(k)*u;return {{"id","dot-forward-error-v1"},{"dtype",dtype},{"K",k},{"absolute_floor",1e-6},{"relative",4*u},{"sum_abs_factor",ku<.5?4*ku/(1-ku):1.0}};}
inline bool valid_dot(double actual,double expected,double sum_abs,const json& policy){return std::isfinite(actual)&&std::abs(actual-expected)<=policy.at("absolute_floor").get<double>()+policy.at("relative").get<double>()*std::abs(expected)+policy.at("sum_abs_factor").get<double>()*sum_abs;}
inline json loaded_modules(){json a=json::array();HMODULE mods[1024];DWORD bytes=0;if(K32EnumProcessModules(GetCurrentProcess(),mods,sizeof(mods),&bytes))for(unsigned i=0;i<std::min<unsigned>(bytes/sizeof(HMODULE),1024);i++){wchar_t path[32768];if(GetModuleFileNameW(mods[i],path,32768)){int n=WideCharToMultiByte(CP_UTF8,0,path,-1,nullptr,0,nullptr,nullptr);std::string p(n,0);WideCharToMultiByte(CP_UTF8,0,path,-1,p.data(),n,nullptr,nullptr);p.pop_back();a.push_back(p);}}return a;}
inline json os_info(){struct Version{ULONG size,major,minor,build,platform;WCHAR service_pack[128];};Version v{};v.size=sizeof(v);using Fn=LONG(WINAPI*)(Version*);auto f=reinterpret_cast<Fn>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"),"RtlGetVersion"));if(!f||f(&v)!=0)return {{"name","Windows"},{"build",nullptr}};return {{"name","Windows"},{"major",v.major},{"minor",v.minor},{"build",v.build},{"architecture","x64"}};}
inline json build_info(){return {{"app_version",GPUBENCH_VERSION},{"protocol_version",1},{"os",os_info()},{"compiler",
#ifdef _MSC_FULL_VER
 std::to_string(_MSC_FULL_VER)
#else
 __VERSION__
#endif
 },{"pointer_bits",sizeof(void*)*8},{"loaded_modules",loaded_modules()}};}
inline std::string case_id(const std::string& suite,const std::string& dtype,const std::string& accumulation="fp32"){if(suite=="alu")return suite+"."+dtype+".fma";if(dtype=="fp32")return suite+".fp32.strict";if(dtype=="fp8")return suite+".fp8.e4m3.acc32";return suite+".fp16."+(accumulation=="fp16"?"acc16":"acc32");}
inline json base_result(const json& device,const json& request,const std::string& backend){auto s=request.value("suite","alu"),d=request.value("dtype","fp32"),acc=s=="gemm"&&d=="fp16"?request.value("accumulation","fp32"):s=="alu"?d:"fp32";return {{"schema_version",1},{"app_version",GPUBENCH_VERSION},{"git_commit",nullptr},{"build",build_info()},{"backend",backend},{"device",device},{"suite",s},{"case_id",case_id(s,d,acc)},{"input_a",d=="fp8"?"ocp_e4m3":d},{"input_b",d=="fp8"?"ocp_e4m3":d},{"accumulation",acc},{"output",s=="alu"?"checksum":acc},{"scaling",d=="fp8"?"tensor":"none"},{"density","dense"},{"implementation","unknown"},{"configuration",request},{"hardware_support","unknown"},{"backend_status","ready"},{"result_status","skipped"},{"reason_code","not_run"},{"reason_message",""},{"validation_status","not_run"},{"reference_kind",nullptr},{"sampled_elements",0},{"error_metrics",json::object()},{"tolerance_policy",json::object()},{"native_evidence",json::array()},{"timing_source",nullptr},{"timing_scope",nullptr},{"sample_times_ms",json::array()},{"raw_samples",json::array()},{"median_ms",nullptr},{"best_ms",nullptr},{"wall_ms",nullptr},{"measurement_duration",0},{"median_tflops",nullptr},{"best_tflops",nullptr}};}
inline json unavailable(json r,std::string hardware,std::string status,std::string code,std::string message){r["hardware_support"]=hardware;r["backend_status"]=status;r["reason_code"]=code;r["reason_message"]=message;r["result_status"]="skipped";r["median_tflops"]=nullptr;r["best_tflops"]=nullptr;return r;}
inline int exit_code(bool cancelled,bool args_error,bool failed,bool require_all,bool unavailable_items,size_t ok){if(cancelled)return 130;if(args_error)return 2;if(failed)return 4;if(require_all&&unavailable_items)return 5;if(!ok)return 3;return 0;}
struct BatchTime {double device_ms;double wall_ms;};
using Progress=std::function<void(const json&)>;
// batch must time the entire ordered GPU interval, excluding external setup/readback.
inline json measure(json r,const json& opt,std::atomic_bool& cancel,uint64_t ops_per_dispatch,const std::function<BatchTime(uint64_t)>& batch,const Progress& progress){
  auto start=Clock::now();uint64_t reps=1;json samples=json::array();
  auto save=[&](){r["raw_samples"]=samples;r["sample_times_ms"]=json::array();double duration=0;for(auto& s:samples){r["sample_times_ms"].push_back(s["elapsed_ms"]);duration+=s["elapsed_ms"].get<double>();}r["measurement_duration"]=duration;r["wall_ms"]=millis(start,Clock::now());r["configuration"]["repetitions"]=reps;};
  try{
    if(r.value("validation_status","")!="passed")throw Error("validation_failed","probe_failed","validation required before formal sampling");
    progress({{"phase","calibration"}});auto t=batch(1);if(!(t.device_ms>0&&std::isfinite(t.device_ms)))throw Error("invalid_timing","probe_failed","invalid device interval");
    if(t.device_ms>100)throw Error("dispatch_budget","configuration_unsupported","single dispatch exceeds 100 ms budget; choose a smaller explicit configuration");
    reps=uint64_t(std::clamp(std::floor(100./t.device_ms),1.,4096.));auto batch_ops=checked_product({ops_per_dispatch,reps});
    r["configuration"]["calibration_ms"]=t.device_ms;r["configuration"]["batch_target_ms"]=100;
    json calibration=json::array();for(int trial=0;trial<4&&!cancel;trial++){auto b=batch(reps);if(!(b.device_ms>0&&std::isfinite(b.device_ms)))throw Error("invalid_timing","probe_failed","invalid calibration interval");calibration.push_back({{"repetitions",reps},{"elapsed_ms",b.device_ms},{"wall_ms",b.wall_ms}});if(b.device_ms<=200||reps==1)break;reps=std::max<uint64_t>(1,uint64_t(std::floor(reps*100./b.device_ms)));}batch_ops=checked_product({ops_per_dispatch,reps});r["configuration"]["batch_calibration"]=calibration;
    double warm=0;progress({{"phase","warmup"}});while(warm<std::max(500.,opt.value("warmup_ms",500.))&&!cancel){auto w=batch(reps);if(!(w.device_ms>0))throw Error("invalid_timing","probe_failed","invalid warmup interval");warm+=w.device_ms;}
    r["configuration"]["actual_warmup_ms"]=warm;double total=0;size_t min_samples=std::max<size_t>(7,opt.value("samples",7));double target=std::max(1000.,opt.value("duration_ms",1000.));
    while(!cancel&&(samples.size()<min_samples||total<target)){auto sample_start=unix_ms();auto w=batch(reps);auto sample_end=unix_ms();if(!(w.device_ms>0&&std::isfinite(w.device_ms)))throw Error("invalid_timing","probe_failed","invalid sample interval");samples.push_back({{"start_unix_ms",sample_start},{"end_unix_ms",sample_end},{"index",samples.size()},{"elapsed_ms",w.device_ms},{"wall_ms",w.wall_ms},{"repetitions",reps},{"flops",batch_ops}});total+=w.device_ms;progress({{"phase","sampling"},{"samples",samples.size()},{"elapsed_ms",total}});if(samples.size()>10000)throw Error("sample_limit","configuration_unsupported","sampling limit exceeded");}
    save();if(cancel){r["result_status"]="cancelled";r["reason_code"]="user_cancelled";return r;}
    std::vector<double> ts;for(auto& s:samples)ts.push_back(s["elapsed_ms"]);double med=median(ts),best=*std::min_element(ts.begin(),ts.end());r["median_ms"]=med;r["best_ms"]=best;r["median_tflops"]=double(batch_ops)/(med*1e9);r["best_tflops"]=double(batch_ops)/(best*1e9);r["result_status"]="ok";r["reason_code"]="ok";r["reason_message"]="";r["noisy"]=(*std::max_element(ts.begin(),ts.end())-best)/med>.1;
  }catch(const Error& e){save();r["result_status"]=e.status=="configuration_unsupported"?"skipped":"failed";r["reason_code"]=e.code;r["reason_message"]=e.what();r["backend_status"]=e.status;}
  return r;
}
}
