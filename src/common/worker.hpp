#pragma once
#include "core.hpp"
#include <iostream>
#include <mutex>
#include <thread>
namespace gb {
struct Backend {
 virtual ~Backend()=default;
 virtual std::string name()const=0;
 virtual json enumerate()=0; // array of devices: id, alias, backend_index, name, vendor_id, device_id, architecture, memory_bytes, identity
 virtual json probe(const json& request)=0;
 virtual json run(const json& request,std::atomic_bool& cancel,const Progress& progress)=0;
};
inline json envelope(const std::string& type,const json& id,const json& payload){return {{"protocol_version",1},{"request_id",id},{"type",type},{"payload",payload}};}
inline int worker_main(Backend& backend){
 SetErrorMode(SEM_FAILCRITICALERRORS|SEM_NOGPFAULTERRORBOX);std::mutex output;std::atomic_bool cancel=false,busy=false;std::thread execution;
 auto send=[&](const std::string& type,const json& id,const json& p){std::lock_guard lock(output);std::cout<<envelope(type,id,p).dump()<<'\n'<<std::flush;};
 auto info=build_info();info["backend"]=backend.name();send("hello",0,info);
 std::string line;while(std::getline(std::cin,line)){
  json id=nullptr;
  try{if(line.size()>16*1024*1024)throw Error("bad_message","probe_failed","message too large");auto msg=json::parse(line);id=msg.at("request_id");if(!(id.is_null()||id.is_string()||id.is_number_integer()))throw Error("bad_message","probe_failed","request_id must be integer, string or null");if(msg.at("protocol_version")!=1||!msg.at("payload").is_object())throw Error("protocol_mismatch","probe_failed","protocol v1 object payload required");auto type=msg.at("type").get<std::string>();auto p=msg["payload"];
   if(type=="cancel"){cancel=true;send("progress",id,{{"phase","cancellation_requested"}});continue;}
   if(type=="shutdown"){cancel=true;break;}
   if(busy)throw Error("worker_busy","configuration_unsupported","only cancel/shutdown accepted during run");
   if(execution.joinable())execution.join();
   if(type=="enumerate")send("devices",id,{{"devices",backend.enumerate()},{"runtime",build_info()}});
   else if(type=="probe")send("capability",id,backend.probe(p));
   else if(type=="run"){cancel=false;busy=true;execution=std::thread([&,p,id](){try{auto result=backend.run(p,cancel,[&](const json& v){send("progress",id,v);});busy=false;send("result",id,result);}catch(const Error& e){busy=false;send("error",id,{{"code",e.code},{"backend_status",e.status},{"message",e.what()}});}catch(const std::exception& e){busy=false;send("error",id,{{"code","execution_error"},{"backend_status","probe_failed"},{"message",e.what()}});}});}
   else throw Error("unknown_command","configuration_unsupported","unknown protocol command");
  }catch(const Error& e){send("error",id,{{"code",e.code},{"backend_status",e.status},{"message",e.what()}});}catch(const std::exception& e){send("error",id,{{"code","bad_message"},{"backend_status","probe_failed"},{"message",e.what()}});}
 }
 cancel=true;if(execution.joinable())execution.join();return 0;
}
}
