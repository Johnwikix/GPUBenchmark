#include "core.hpp"
#include <iostream>
using namespace gb;
static void require(bool v,const char* m){if(!v)throw std::runtime_error(m);}
int main(){try{
 auto acc16=base_result({{"id","test"}},{{"suite","gemm"},{"dtype","fp16"},{"accumulation","fp16"}},"cuda");require(acc16["case_id"]=="gemm.fp16.acc16"&&acc16["accumulation"]=="fp16"&&acc16["output"]=="fp16","acc16 identity and output");require(case_id("gemm","fp16")=="gemm.fp16.acc32"&&case_id("gemm","fp8","fp16")=="gemm.fp8.e4m3.acc32","acc32 identities preserved");
 require(checked_product({2,3,4,5})==120,"FLOP product");bool overflow=false;try{checked_product({UINT64_MAX,2});}catch(...){overflow=true;}require(overflow,"overflow detection");
 require(median({7,1,3})==3&&median({1,2,3,4})==2.5,"median");require(half_decode(0x3c00)==1&&half_decode(0xc000)==-2&&half_decode(1)==std::ldexp(1.,-24),"half decode");
 require(e4m3_decode(0x38)==1&&e4m3_decode(0x7e)==448&&std::isnan(e4m3_decode(0x7f)),"E4M3 decode");for(int i=0;i<256;i++)if(std::isfinite(e4m3_decode(uint8_t(i))))require(e4m3_decode(e4m3_encode(e4m3_decode(uint8_t(i))))==e4m3_decode(uint8_t(i)),"E4M3 roundtrip");require(e4m3_encode(1.0625)==0x38,"ties to even");
 std::vector<double>a={1,2,3,4,5,6},b={2,0,1,-1,3,2};require(reference_dot(a,b,2,2,3,1,0)==31,"non-square row major reference");auto policy=tolerance_policy("fp32",3);require(valid_dot(31,31,31,policy)&&!valid_dot(17,31,31,policy),"layout failure rejected");
 require(exit_code(true,true,true,true,true,1)==130&&exit_code(false,true,true,true,true,1)==2&&exit_code(false,false,true,true,true,1)==4&&exit_code(false,false,false,true,true,1)==5&&exit_code(false,false,false,false,true,0)==3&&exit_code(false,false,false,false,true,1)==0,"exit precedence");
 auto r=base_result({{"id","test"}},{{"suite","alu"},{"dtype","fp32"}},"test");require(r["median_tflops"].is_null(),"unavailable null");std::atomic_bool cancel=false;auto fail=measure(r,json::object(),cancel,128,[](uint64_t){return BatchTime{10,10};},[](const json&){});require(fail["result_status"]=="failed"&&fail["median_tflops"].is_null(),"invalid results do not publish throughput");
 std::cout<<"common: passed (no GPU performance generated)\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
