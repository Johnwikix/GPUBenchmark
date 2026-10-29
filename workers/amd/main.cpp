#include "amd.hpp"
namespace amd {
json alu(json r,const json& opt,std::atomic_bool& cancel,const gb::Progress& progress){
  const bool h=opt.value("dtype","fp32")=="fp16";const unsigned lanes=h?2:1,seed=opt.value("seed",20261020u);
  hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,r["device"].at("backend_index")),"device properties");
  Queue q;unsigned iterations=128,selectedA=4,selectedU=4,selectedT=64,selectedBlocks=std::max(1,prop.multiProcessorCount)*4;
  double best=0;json search=json::array();
  // Each candidate is a separate tuning dispatch, never part of formal samples.
  for(unsigned a:{4u,8u,16u})for(unsigned t:{64u,128u,256u})for(unsigned u:{4u,8u}){
    if(cancel)break;
    size_t bytes=gb::checked_product({selectedBlocks,t,a,lanes,4});if(bytes>budget())continue;
    Buffer out(bytes);auto tm=q.time(1,[&](uint64_t){check(gb_amd_alu_launch(h,a,u,selectedBlocks,t,iterations,seed,out.p,q.stream),"ALU tune");});
    double rate=double(gb::checked_product({selectedBlocks,t,a,lanes,iterations,2}))/tm.device_ms;
    search.push_back({{"accumulators",a},{"workgroup_size",t},{"unroll",u},{"iterations",iterations},{"elapsed_ms",tm.device_ms}});
    if(rate>best){best=rate;selectedA=a;selectedU=u;selectedT=t;}
  }
  if(cancel){r["result_status"]="cancelled";r["reason_code"]="user_cancelled";return r;}
  const uint64_t count=gb::checked_product({selectedBlocks,selectedT,selectedA,lanes});size_t bytes=gb::checked_product({count,4});
  unsigned groupCount=groups(opt);if(gb::checked_product({bytes,groupCount})>budget())throw gb::Error("memory_budget","configuration_unsupported","ALU buffers exceed half the currently free memory");
  std::vector<std::unique_ptr<Buffer>> outputs;for(unsigned g=0;g<groupCount;g++)outputs.push_back(std::make_unique<Buffer>(bytes));
  auto launch=[&](unsigned it,uint64_t index){check(gb_amd_alu_launch(h,selectedA,selectedU,selectedBlocks,selectedT,it,seed,outputs[index%groupCount]->p,q.stream),"ALU launch");};
  // Validate every channel for a short recurrence, using FP16-rounded CPU
  // arithmetic for the half accumulator contract, then full output finite checks.
  q.time(1,[&](uint64_t i){launch(16,i);});auto shortValues=outputs[0]->floats();double maxError=0;
  for(uint64_t tid=0;tid<uint64_t(selectedBlocks)*selectedT;tid++)for(unsigned a=0;a<selectedA;a++)for(unsigned lane=0;lane<lanes;lane++){
    float x=(1+((unsigned(tid)+a+seed+lane*3)%31))*.015625f,b=(1+((seed+unsigned(tid))%7))*.015625f+float(a+lane)*.001953125f;
    for(unsigned i=0;i<16;i++)x=h?round_half(std::fma(x,-.99951171875f,b)):std::fma(x,-.99951171875f,b);
    double error=std::abs(double(x)-shortValues[(tid*selectedA+a)*lanes+lane]);maxError=std::max(maxError,error);
    if(!std::isfinite(error)||error>(h?0.:2e-6))throw gb::Error("validation_failed","probe_failed","ALU short CPU reference mismatch");
  }
  auto t=q.time(1,[&](uint64_t i){launch(iterations,i);});
  // Limit loop growth and dispatch duration. The exact rounded iteration count
  // is fixed before warmup and shared by all formal samples.
  iterations=unsigned(std::clamp(std::floor(5./t.device_ms)*iterations,double(selectedU),1048576.));iterations=std::max(selectedU,iterations/selectedU*selectedU);
  auto full=q.time(1,[&](uint64_t i){launch(iterations,i);});if(full.device_ms>20)throw gb::Error("dispatch_budget","configuration_unsupported","calibrated ALU dispatch exceeds 20 ms");
  finite(outputs[0]->floats(),r);
  r=ready(r);r["validation_status"]="passed";r["reference_kind"]="cpu_fma_short_full_and_gpu_full_finite";r["sampled_elements"]=count;
  r["error_metrics"]["short_max_abs_error"]=maxError;r["tolerance_policy"]={{"id","alu-recurrence-v1"},{"short_iterations",16},{"absolute",h?0.:2e-6}};
  r["native_evidence"].push_back({{"kind","documented_api_path"},{"detail",h?"HIP __hfma2; packed half accumulation; ISA validation pending":"HIP fmaf; FP32 accumulation; ISA validation pending"}});
  r["configuration"].update({{"threads",uint64_t(selectedBlocks)*selectedT},{"workgroup_size",selectedT},{"independent_accumulators",selectedA},{"lanes_per_instruction",lanes},{"iterations",iterations},{"unroll",selectedU},{"kernel_variant",h?"hip_hfma2":"hip_fmaf"},{"tuning_candidates",search},{"buffer_groups",groupCount},{"allocated_bytes",bytes*groupCount}});
  uint64_t cursor=0,ops=gb::checked_product({selectedBlocks,selectedT,selectedA,lanes,iterations,2});
  r=gb::measure(r,opt,cancel,ops,[&](uint64_t reps){return q.time(reps,[&](uint64_t){launch(iterations,cursor++);});},progress);
  if(!cancel)try{for(auto& output:outputs)finite(output->floats(),r);}catch(const gb::Error& e){return failed(r,e);}return r;
}
struct Module {
  hipModule_t value=nullptr;
  explicit Module(const std::filesystem::path& path){if(!std::filesystem::is_regular_file(path))throw gb::Error("code_object_missing","kernel_missing","missing WMMA code object: "+path.string());check(hipModuleLoad(&value,path.string().c_str()),"WMMA module load");}
  ~Module(){if(value)hipModuleUnload(value);}
};
json micro(json r,const json& opt,std::atomic_bool& cancel,const gb::Progress& progress){
  const auto arch=architecture(r["device"]);const bool fp8=opt.value("dtype","fp16")=="fp8";
  if(!rdna3(arch)&&!rdna4(arch))return gb::unavailable(r,"unknown","kernel_missing","architecture_not_targeted","direct WMMA currently has code objects only for gfx1100/1101/1102/1200/1201");
  Module mod(directory()/"kernels"/(arch+".hsaco"));Queue q;hipDeviceProp_t prop{};check(hipGetDeviceProperties(&prop,r["device"]["backend_index"]),"properties");
  unsigned blocks=std::max(1,prop.multiProcessorCount)*4,threads=128,acc=2,iterations=64;
  const uint64_t waves=uint64_t(blocks)*threads/32;const unsigned groupCount=groups(opt),seed=opt.value("seed",20261020u);
  // Dyadic nonzero signed inputs give a strict exact short-loop test without
  // layout ambiguities. A and B vary along both axes and between accumulators.
  std::vector<double> refA(4*256),refB(4*256);std::vector<uint16_t> hA(4*256),hB(4*256);std::vector<uint8_t> fA(4*256),fB(4*256);
  std::mt19937 random(seed);for(size_t i=0;i<refA.size();i++){
    float a=(int(random()%15)-7)*.03125f,b=(int(random()%15)-7)*.03125f;
    if(fp8){fA[i]=gb::e4m3_encode(a);fB[i]=gb::e4m3_encode(b);refA[i]=gb::e4m3_decode(fA[i]);refB[i]=gb::e4m3_decode(fB[i]);}
    else{hA[i]=half_encode(a);hB[i]=half_encode(b);refA[i]=gb::half_decode(hA[i]);refB[i]=gb::half_decode(hB[i]);}
  }
  Buffer inputA(fp8?fA.size():hA.size()*2),inputB(fp8?fB.size():hB.size()*2);if(fp8){inputA.upload(fA);inputB.upload(fB);}else{inputA.upload(hA);inputB.upload(hB);}
  size_t bytes=gb::checked_product({waves,4,256,4});if(gb::checked_product({bytes,groupCount})+inputA.bytes+inputB.bytes>budget())throw gb::Error("memory_budget","configuration_unsupported","WMMA buffers exceed half the currently free memory");
  std::vector<std::unique_ptr<Buffer>> outputs;for(unsigned i=0;i<groupCount;i++)outputs.push_back(std::make_unique<Buffer>(bytes));
  hipFunction_t function=nullptr;auto setFunction=[&](){const std::string symbol="gb_wmma_"+std::string(fp8?"fp8":"fp16")+"_a"+std::to_string(acc);check(hipModuleGetFunction(&function,mod.value,symbol.c_str()),"WMMA function");};
  auto launch=[&](unsigned it,uint64_t index){void* a=inputA.p;void* b=inputB.p;void* out=outputs[index%groupCount]->p;void* args[]={&a,&b,&out,&it};check(hipModuleLaunchKernel(function,blocks,1,1,threads,1,1,0,q.stream,args,nullptr),"WMMA launch");};
  json search=json::array();double fastest=0;unsigned bestAcc=2,bestThreads=128;
  for(unsigned candidate:{2u,4u})for(unsigned t:{64u,128u,256u}){
    if(cancel)break;acc=candidate;threads=t; // Reallocate for the largest grid before timing.
    size_t need=gb::checked_product({uint64_t(blocks)*256/32,4,256,4});
    if(need*groupCount+inputA.bytes+inputB.bytes>budget())continue;
    for(auto& b:outputs)if(b->bytes<need)b=std::make_unique<Buffer>(need);
    setFunction();auto tm=q.time(1,[&](uint64_t i){launch(iterations,i);});double rate=double(blocks)*threads/32*acc*8192*iterations/tm.device_ms;
    search.push_back({{"accumulators",acc},{"workgroup_size",threads},{"elapsed_ms",tm.device_ms},{"iterations",iterations}});if(rate>fastest){fastest=rate;bestAcc=acc;bestThreads=threads;}
  }
  acc=bestAcc;threads=bestThreads;setFunction();uint64_t actualWaves=uint64_t(blocks)*threads/32,count=actualWaves*acc*256;
  q.time(1,[&](uint64_t i){launch(3,i);});auto values=outputs[0]->floats();double maxError=0;
  std::vector<double> expected(acc*256);for(unsigned s=0;s<acc;s++)for(unsigned row=0;row<16;row++)for(unsigned col=0;col<16;col++){
    long double sum=0;for(unsigned k=0;k<16;k++)sum+=refA[s*256+row*16+k]*refB[s*256+k*16+col];expected[s*256+row*16+col]=double(sum)*3;
  }
  for(uint64_t i=0;i<count;i++){double err=std::abs(values[i]-expected[i%(acc*256)]);maxError=std::max(maxError,err);if(!std::isfinite(err)||err!=0)throw gb::Error("validation_failed","probe_failed","WMMA short complete tile CPU reference mismatch; check generation-specific fragment mapping");}
  auto tm=q.time(1,[&](uint64_t i){launch(iterations,i);});iterations=unsigned(std::clamp(std::floor(5./tm.device_ms)*iterations,1.,65536.));
  auto full=q.time(1,[&](uint64_t i){launch(iterations,i);});if(full.device_ms>20)throw gb::Error("dispatch_budget","configuration_unsupported","WMMA dispatch exceeds 20 ms");
  values=outputs[0]->floats();values.resize(size_t(count));finite(values,r);
  r=ready(r);r["scaling"]="none";r["validation_status"]="passed";r["reference_kind"]="cpu_fp64_complete_tiles_short_and_gpu_full_finite";r["sampled_elements"]=count;
  r["error_metrics"]["short_max_abs_error"]=maxError;r["tolerance_policy"]={{"id","wmma-dyadic-exact-v1"},{"short_iterations",3},{"absolute",0}};
  r["native_evidence"].push_back({{"kind","documented_api_path"},{"detail","AMD wave32 WMMA builtin; disassembly and hardware certification pending"}});
  r["configuration"].update({{"kernel_variant",std::string(rdna3(arch)?"rdna3_":"rdna4_")+(fp8?"fp8":"fp16")},{"code_object",(directory()/"kernels"/(arch+".hsaco")).string()},{"threads",uint64_t(blocks)*threads},{"workgroup_size",threads},{"wave_size",32},{"wave_count",actualWaves},{"instruction_tile",{16,16,16}},{"flops_per_wave_instruction",8192},{"iterations",iterations},{"independent_accumulators",acc},{"buffer_groups",groupCount},{"tuning_candidates",search}});
  uint64_t cursor=0;auto ops=gb::checked_product({actualWaves,acc,8192,iterations});
  r=gb::measure(r,opt,cancel,ops,[&](uint64_t reps){return q.time(reps,[&](uint64_t){launch(iterations,cursor++);});},progress);
  if(!cancel)try{for(auto& output:outputs){values=output->floats();values.resize(size_t(count));finite(values,r);}}catch(const gb::Error& e){return failed(r,e);}return r;
}
class Backend final:public gb::Backend {
 public:
  std::string name()const override{return "amd";}
  json enumerate()override{
    int count=0;check(hipGetDeviceCount(&count),"HIP enumerate");json devices=json::array();int runtime=0,driver=0;check(hipRuntimeGetVersion(&runtime),"runtime version");check(hipDriverGetVersion(&driver),"driver version");
    for(int i=0;i<count;i++){
      hipDeviceProp_t p{};check(hipGetDeviceProperties(&p,i),"device properties");std::string name=p.name;
      // Reject explicitly out-of-scope product families, while capability probes
      // stay architecture-based for unknown/new consumer marketing names.
      if(name.find("Instinct")!=std::string::npos||name.find("Radeon PRO")!=std::string::npos||name.find("Radeon Pro")!=std::string::npos||name.find("Radeon AI PRO")!=std::string::npos)continue;
      char pci[64]{};json identity=json::object();if(hipDeviceGetPCIBusId(pci,sizeof(pci),i)==hipSuccess)identity["pci_bus_id"]=pci;
      auto hexBytes=[](const char* bytes,size_t size){std::ostringstream value;bool nonzero=false;for(size_t j=0;j<size;j++){unsigned byte=static_cast<unsigned char>(bytes[j]);nonzero|=byte!=0;value<<std::hex<<std::setw(2)<<std::setfill('0')<<byte;}return nonzero?value.str():std::string{};};
      auto uuid=hexBytes(p.uuid.bytes,16),luid=hexBytes(p.luid,8);if(!uuid.empty())identity["uuid"]=uuid;if(!luid.empty()){identity["luid"]=luid;identity["luid_device_node_mask"]=p.luidDeviceNodeMask;}
      devices.push_back({{"id","amd:"+std::to_string(i)},{"alias","amd:"+std::to_string(i)},{"backend_index",i},{"name",name},{"vendor_id",0x1002},{"device_id",nullptr},{"architecture",p.gcnArchName},{"memory_bytes",p.totalGlobalMem},{"integrated",p.integrated!=0},{"identity",identity},{"runtime_version",runtime},{"driver_version",driver},{"certification","not_hardware_validated"}});
    }return devices;
  }
  json probe(const json& req)override{
    const auto d=req.at("device");check(hipSetDevice(d.at("backend_index")),"select device");Queue q;Buffer output(64*4*4);
    q.time(1,[&](uint64_t){check(gb_amd_alu_launch(0,4,4,1,64,16,20261020,output.p,q.stream),"short kernel probe");});json v=json::object();finite(output.floats(),v);
    json cases=json::array();for(const std::string suite:{"alu","gemm","matrix_micro"})for(const std::string dtype:{"fp32","fp16","fp8"}){
      std::string hw=hardware(d,dtype),status="ready",reason="run performs configuration-specific numerical validation";
      if((suite=="alu"&&dtype=="fp8")||(suite=="matrix_micro"&&dtype=="fp32")){status="api_unsupported";reason="case is outside implemented suite semantics";}
      else if(hw=="unsupported"){status="api_unsupported";reason="architecture lacks this native precision";}
      else if(suite=="matrix_micro"&&!std::filesystem::is_regular_file(directory()/"kernels"/(architecture(d)+".hsaco"))){status="kernel_missing";reason="matching architecture code object absent";}
      cases.push_back({{"suite",suite},{"dtype",dtype},{"hardware_support",hw},{"backend_status",status},{"reason_message",reason},{"execution_verified",false}});
    }
    return {{"device",d},{"short_fp32_kernel_probe","passed"},{"cases",cases},{"optional_libraries",blas_info()},{"loaded_modules",gb::loaded_modules()},{"effective_environment",effective_environment()},{"certification","not_hardware_validated"}};
  }
  json run(const json& opt,std::atomic_bool& cancel,const gb::Progress& progress)override{
    auto r=gb::base_result(opt.at("device"),opt,"amd");r["hardware_support"]=hardware(r["device"],opt.value("dtype","fp32"));
    try{
      if(device_lost)throw gb::Error("device_lost","driver_incompatible","prior HIP device loss; restart worker after resolving driver/device state");
      const auto dtype=opt.value("dtype","fp32"),suite=opt.value("suite","alu");if(dtype!="fp32"&&dtype!="fp16"&&dtype!="fp8")return gb::unavailable(r,"unknown","api_unsupported","dtype_not_implemented","initial release accepts fp32/fp16/fp8 only");
      if(r["hardware_support"]=="unsupported")return gb::unavailable(r,"unsupported","api_unsupported","native_precision_unsupported","architecture lacks the requested native arithmetic");
      check(hipSetDevice(opt.at("device").at("backend_index")),"select device");
      if(suite=="alu"&&(dtype=="fp32"||dtype=="fp16"))return alu(r,opt,cancel,progress);
      if(suite=="matrix_micro"&&(dtype=="fp16"||dtype=="fp8"))return micro(r,opt,cancel,progress);
      if(suite=="gemm")return gemm(r,opt,cancel,progress);
      return gb::unavailable(r,r["hardware_support"],"api_unsupported","case_not_implemented","suite/dtype combination is not implemented");
    }catch(const gb::Error& e){return failed(r,e);}
  }
};
}
int main(){
  // Process-local settings are set before explicitly loading math DLLs. HIP
  // runtime registration itself precedes main and must be deployed correctly.
  SetEnvironmentVariableW(L"HIPBLASLT_LOG_LEVEL",L"0");SetEnvironmentVariableW(L"HIPBLASLT_LOG_MASK",L"0");SetEnvironmentVariableW(L"HIPBLASLT_LOG_FILE",L"NUL");
  SetEnvironmentVariableW(L"HIPBLASLT_OVERRIDE_COMPUTE_TYPE_XF32",L"0");SetEnvironmentVariableW(L"ROCBLAS_LAYER",L"0");
  amd::Backend backend;return gb::worker_main(backend);
}
