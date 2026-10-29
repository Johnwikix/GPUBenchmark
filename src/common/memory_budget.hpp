#pragma once
#include "core.hpp"
#include <dxgi1_4.h>
namespace gb {
inline json dxgi_budget(const std::vector<unsigned char>& luid_bytes,uint32_t vendor_id){
 IDXGIFactory1* factory=nullptr;json result={{"source","unavailable"},{"available_bytes",nullptr},{"allocation_budget_bytes",nullptr}};if(FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1),reinterpret_cast<void**>(&factory))))return result;
 IDXGIAdapter1* match=nullptr;unsigned matches=0;
 for(UINT i=0;;i++){IDXGIAdapter1* adapter=nullptr;if(factory->EnumAdapters1(i,&adapter)==DXGI_ERROR_NOT_FOUND)break;if(!adapter)break;DXGI_ADAPTER_DESC1 desc{};adapter->GetDesc1(&desc);bool same=luid_bytes.size()==8?memcmp(&desc.AdapterLuid,luid_bytes.data(),8)==0:desc.VendorId==vendor_id;
  if(same&&!(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)){matches++;if(!match)match=adapter;else adapter->Release();}else adapter->Release();
 }factory->Release();if(match&&matches==1){IDXGIAdapter3* a3=nullptr;if(SUCCEEDED(match->QueryInterface(__uuidof(IDXGIAdapter3),reinterpret_cast<void**>(&a3)))){DXGI_QUERY_VIDEO_MEMORY_INFO info{};if(SUCCEEDED(a3->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info))){uint64_t available=info.Budget>info.CurrentUsage?info.Budget-info.CurrentUsage:0;result={{"source",luid_bytes.size()==8?"dxgi_luid":"dxgi_unique_vendor_adapter"},{"available_bytes",available},{"allocation_budget_bytes",available/2},{"budget_bytes",info.Budget},{"current_usage_bytes",info.CurrentUsage}};}a3->Release();}}
 if(match)match->Release();return result;
}
inline uint16_t half_encode(double x){if(std::isnan(x))return 0x7e00;uint16_t sign=std::signbit(x)?0x8000:0;double a=std::abs(x);if(a>=65520||std::isinf(a))return sign|0x7c00;if(a<std::ldexp(1.,-14))return sign|uint16_t(std::nearbyint(std::ldexp(a,24)));int e=0;std::frexp(a,&e);e--;auto m=int(std::nearbyint((std::ldexp(a,-e)-1)*1024));if(m==1024){m=0;e++;}return sign|uint16_t((e+15)<<10)|uint16_t(m);}
}
