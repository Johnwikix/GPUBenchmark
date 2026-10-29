#pragma once
#include "worker.hpp"
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <cstdio>
namespace gb {
inline std::atomic_bool interrupted=false;
inline BOOL WINAPI ctrl_handler(DWORD type){if(type==CTRL_C_EVENT||type==CTRL_BREAK_EVENT){interrupted=true;return TRUE;}return FALSE;}
// Live progress is one in-place \r line on stderr; the full progress stream
// stays in the per-worker protocol transcript. progress_clear() lets the host
// wipe the line before printing a result row or a section header.
inline std::atomic<size_t> progress_line_width{0};
inline void progress_clear(){if(size_t n=progress_line_width.exchange(0))std::cerr<<"\r"<<std::string(n,' ')<<"\r";}
inline void progress_show(const std::string& line){auto old=progress_line_width.exchange(line.size());std::cerr<<'\r'<<line<<std::string(old>line.size()?old-line.size():0,' ')<<std::flush;}
inline std::string progress_text(const json& p){std::string s=p.value("phase","progress");char buf[64];for(auto& [k,v]:p.items()){if(k=="phase")continue;if(v.is_number_float()){std::snprintf(buf,sizeof buf,"%.1f",v.get<double>());s+=" "+k+"="+buf;}else if(v.is_number_integer())s+=" "+k+"="+std::to_string(v.get<int64_t>());else if(v.is_boolean())s+=" "+k+(v.get<bool>()?"=true":"=false");}return s;}
inline std::wstring wide(const std::string& s){if(s.empty())return {};int n=MultiByteToWideChar(CP_UTF8,MB_ERR_INVALID_CHARS,s.data(),int(s.size()),nullptr,0);if(!n)throw std::runtime_error("invalid UTF-8");std::wstring w(n,0);MultiByteToWideChar(CP_UTF8,0,s.data(),int(s.size()),w.data(),n);return w;}
inline std::string utf8(const std::wstring& w){if(w.empty())return {};int n=WideCharToMultiByte(CP_UTF8,0,w.data(),int(w.size()),nullptr,0,nullptr,nullptr);std::string s(n,0);WideCharToMultiByte(CP_UTF8,0,w.data(),int(w.size()),s.data(),n,nullptr,nullptr);return s;}
struct Handle {HANDLE h=nullptr;Handle()=default;explicit Handle(HANDLE v):h(v){}~Handle(){if(h&&h!=INVALID_HANDLE_VALUE)CloseHandle(h);}Handle(const Handle&)=delete;Handle& operator=(const Handle&)=delete;operator HANDLE()const{return h;}void reset(HANDLE v=nullptr){if(h&&h!=INVALID_HANDLE_VALUE)CloseHandle(h);h=v;}};
inline json logging_environment(const std::filesystem::path& log){return {{"HIPBLASLT_LOG_LEVEL","0"},{"HIPBLASLT_LOG_MASK","0"},{"HIPBLASLT_LOG_FILE",utf8(log.wstring())},{"ROCBLAS_LAYER","0"},{"ROCBLAS_LOG_TRACE_PATH",utf8(log.wstring())},{"CUBLAS_LOGINFO_DBG","0"},{"CUBLAS_LOGDEST_DBG",utf8(log.wstring())},{"CUBLASLT_LOG_LEVEL","0"},{"CUBLASLT_LOG_MASK","0"},{"CUBLASLT_LOG_FILE",utf8(log.wstring())},{"NVIDIA_TF32_OVERRIDE","0"},{"CUBLAS_EMULATE_SINGLE_PRECISION","0"},{"CUBLAS_LOGTRACE_DBG","0"},{"MKL_VERBOSE","0"},{"ONEMKL_VERBOSE","0"},{"DNNL_VERBOSE","0"},{"SYCL_PI_TRACE","0"},{"UR_LOG_LOADER","level:quiet"}};}
struct EnvLess {bool operator()(const std::wstring& a,const std::wstring& b)const{return _wcsicmp(a.c_str(),b.c_str())<0;}};
inline std::vector<wchar_t> child_environment(const json& overrides){std::map<std::wstring,std::wstring,EnvLess> env;wchar_t* block=GetEnvironmentStringsW();if(!block)throw std::runtime_error("GetEnvironmentStrings failed");for(auto p=block;*p;p+=wcslen(p)+1){std::wstring line=p;auto eq=line.find(L'=',line[0]==L'='?1:0);if(eq!=line.npos)env[line.substr(0,eq)]=line.substr(eq+1);}FreeEnvironmentStringsW(block);for(auto& [k,v]:overrides.items())env[wide(k)]=wide(v.get<std::string>());std::vector<wchar_t> b;for(auto& [k,v]:env){auto s=k+L"="+v;b.insert(b.end(),s.begin(),s.end());b.push_back(0);}b.push_back(0);return b;}
class Worker {
 Handle input,output,process,thread,job,errors;
 std::string pending;uint64_t serial=0;std::ofstream transcript;bool cancellation_sent=false;Clock::time_point cancellation_start{};
 [[noreturn]] void exited(){WaitForSingleObject(process,100);DWORD code=0;GetExitCodeProcess(process,&code);auto status=code==0xC0000135?"runtime_missing":code==0xC000007B?"driver_incompatible":"probe_failed";throw Error(code==0xC0000135?"dependent_dll_missing":code==0xC000007B?"dll_architecture_mismatch":"worker_crashed",status,"worker exited/closed pipe: "+std::to_string(code)+"; inspect stderr log (loader failure can precede main)");}
 void write(const json& message){auto s=message.dump()+"\n";transcript<<"{\"direction\":\"host_to_worker\",\"message\":"<<message.dump()<<"}\n"<<std::flush;DWORD n=0;if(!WriteFile(input,s.data(),DWORD(s.size()),&n,nullptr)||n!=s.size())throw Error("worker_pipe_closed","probe_failed","worker stdin is closed");}
 json next(uint64_t timeout_ms,bool cancellable){auto started=Clock::now();while(true){
   if(cancellable&&interrupted&&!cancellation_sent){write(envelope("cancel",++serial,json::object()));cancellation_sent=true;cancellation_start=Clock::now();}
   if(cancellation_sent&&millis(cancellation_start,Clock::now())>5000){TerminateJobObject(job,130);throw Error("user_cancelled","probe_failed","worker did not cancel within 5 seconds");}
   auto end=pending.find('\n');if(end!=pending.npos){auto line=pending.substr(0,end);pending.erase(0,end+1);json m;try{m=json::parse(line);}catch(...){throw Error("bad_worker_message","probe_failed","non-JSON data on worker stdout");}transcript<<"{\"direction\":\"worker_to_host\",\"message\":"<<m.dump()<<"}\n"<<std::flush;if(!m.is_object()||m.value("protocol_version",0)!=1||!m.contains("request_id")||!m.contains("payload")||!m["payload"].is_object()||!m.contains("type")||!m["type"].is_string())throw Error("bad_worker_message","probe_failed","invalid protocol envelope");return m;}
   DWORD bytes=0;if(!PeekNamedPipe(output,nullptr,0,nullptr,&bytes,nullptr))exited();
   if(bytes){char b[8192];DWORD n=0;if(!ReadFile(output,b,std::min<DWORD>(bytes,sizeof(b)),&n,nullptr))throw Error("worker_crashed","probe_failed","worker output read failed");pending.append(b,n);if(pending.size()>64*1024*1024)throw Error("bad_worker_message","probe_failed","worker message exceeds 64 MiB");continue;}
   if(WaitForSingleObject(process,0)==WAIT_OBJECT_0)exited();
   if(millis(started,Clock::now())>timeout_ms){TerminateJobObject(job,4);throw Error("worker_timeout","probe_failed","worker response timeout");}Sleep(10);
  }}
public:
 json hello,effective_environment;std::string name;
 Worker(std::string backend,const std::filesystem::path& exe,const std::filesystem::path& logdir):name(std::move(backend)){
  std::filesystem::create_directories(logdir);transcript.open(logdir/(name+".protocol.jsonl"),std::ios::app);if(!transcript)throw Error("output_write_failed","probe_failed","cannot open protocol transcript");effective_environment=logging_environment(std::filesystem::absolute(logdir/(name+".library.log")));auto env=child_environment(effective_environment);
  SECURITY_ATTRIBUTES sa{sizeof(sa),nullptr,TRUE};HANDLE read_in=nullptr,write_in=nullptr,read_out=nullptr,write_out=nullptr;
  if(!CreatePipe(&read_in,&write_in,&sa,0))throw std::runtime_error("CreatePipe stdin failed");Handle child_in(read_in);input.reset(write_in);
  if(!CreatePipe(&read_out,&write_out,&sa,0))throw std::runtime_error("CreatePipe stdout failed");output.reset(read_out);Handle child_out(write_out);SetHandleInformation(input,HANDLE_FLAG_INHERIT,0);SetHandleInformation(output,HANDLE_FLAG_INHERIT,0);
  errors.reset(CreateFileW((logdir/(name+".stderr.log")).c_str(),FILE_APPEND_DATA,FILE_SHARE_READ|FILE_SHARE_WRITE,&sa,OPEN_ALWAYS,FILE_ATTRIBUTE_NORMAL,nullptr));if(errors.h==INVALID_HANDLE_VALUE)throw std::runtime_error("cannot open worker stderr");
  job.reset(CreateJobObjectW(nullptr,nullptr));JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};limits.BasicLimitInformation.LimitFlags=JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;if(!job.h||!SetInformationJobObject(job,JobObjectExtendedLimitInformation,&limits,sizeof(limits)))throw std::runtime_error("Job Object creation failed");
  STARTUPINFOW si{};si.cb=sizeof(si);si.dwFlags=STARTF_USESTDHANDLES;si.hStdInput=child_in;si.hStdOutput=child_out;si.hStdError=errors;PROCESS_INFORMATION pi{};auto command=L"\""+exe.wstring()+L"\"";
  if(!CreateProcessW(exe.c_str(),command.data(),nullptr,nullptr,TRUE,CREATE_SUSPENDED|CREATE_UNICODE_ENVIRONMENT|CREATE_NO_WINDOW,env.data(),exe.parent_path().c_str(),&si,&pi))throw Error("worker_launch_failed","runtime_missing","CreateProcess failed: "+std::to_string(GetLastError()));process.reset(pi.hProcess);thread.reset(pi.hThread);
  if(!AssignProcessToJobObject(job,process)){TerminateProcess(process,4);throw std::runtime_error("AssignProcessToJobObject failed");}ResumeThread(thread);child_in.reset();child_out.reset();hello=next(15000,false);if(hello["type"]!="hello")throw Error("bad_worker_message","probe_failed","worker must send hello first");
 }
 ~Worker(){try{write(envelope("shutdown",++serial,json::object()));WaitForSingleObject(process,2000);}catch(...){} }
 json request(const std::string& type,const json& payload,uint64_t timeout=30000){cancellation_sent=false;auto id=++serial;write(envelope(type,id,payload));std::string label;if(payload.contains("suite")){label=payload.value("suite","")+"."+payload.value("dtype","");if(payload.value("m",uint64_t(0)))label+=" @"+std::to_string(payload.at("m").get<uint64_t>());}auto start=Clock::now();while(true){auto elapsed=millis(start,Clock::now());if(elapsed>=timeout){progress_clear();TerminateJobObject(job,4);throw Error("worker_timeout","probe_failed","request deadline exceeded");}auto m=next(timeout-uint64_t(elapsed),type=="run");auto t=m["type"].get<std::string>();if(t=="progress"){if(type=="run"&&!label.empty())progress_show("  ["+name+"] "+label+": "+progress_text(m["payload"]));continue;}if(m["request_id"]!=id){progress_clear();throw Error("bad_worker_message","probe_failed","unexpected request_id");}if(t=="error"){auto p=m["payload"];progress_clear();throw Error(p.value("code","worker_error"),p.value("backend_status","probe_failed"),p.value("message","worker error"));}auto expected=type=="enumerate"?"devices":type=="probe"?"capability":"result";if(t!=expected){progress_clear();throw Error("bad_worker_message","probe_failed","unexpected response type");}return m["payload"];}}
};
}
