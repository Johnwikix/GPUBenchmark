#include "sm120_api.hpp"
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
int main(int argc,char** argv){
 int v=argc>1?std::atoi(argv[1]):4,n=argc>2?std::atoi(argv[2]):128;
 if(n<128||n>4096||n%128)return 2;
 void *a=nullptr,*b=nullptr,*c=nullptr,*d=nullptr,*w=nullptr,*op=nullptr;
 const bool padded=argc>3&&std::string(argv[3])=="padded";cudaStream_t stream=nullptr;
 auto check=[](cudaError_t s){if(s!=cudaSuccess){std::cerr<<cudaGetErrorName(s)<<": "<<cudaGetErrorString(s)<<'\n';std::exit(4);}};
 check(cudaMalloc(&a,size_t(n)*n));check(cudaMalloc(&b,size_t(n)*n));check(cudaMalloc(&c,size_t(n)*n*4));check(cudaMalloc(&d,size_t(n)*n*4));check(cudaMalloc(&w,256ull*1024*1024));
 check(cudaMemset(a,0x20,size_t(n)*n));check(cudaMemset(b,0x20,size_t(n)*n));check(cudaMemset(c,0,size_t(n)*n*4));
 if(padded){check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));std::vector<unsigned char> x(size_t(n)*n,0),y(x.size(),0);for(int row=0;row<17;row++)for(int k=0;k<19;k++)x[size_t(row)*n+k]=0x20;for(int col=0;col<23;col++)for(int k=0;k<19;k++)y[size_t(col)*n+k]=0x20;check(cudaMemcpy(a,x.data(),x.size(),cudaMemcpyHostToDevice));check(cudaMemcpy(b,y.data(),y.size(),cudaMemcpyHostToDevice));}
 check(cuda_bench::sm120_create(v,n,n,n,a,b,c,d,.75f,w,padded?16ull*1024*1024:256ull*1024*1024,stream,&op));
 check(cuda_bench::sm120_launch(op,stream));check(cudaDeviceSynchronize());
 std::vector<float> result(size_t(n)*n);check(cudaMemcpy(result.data(),d,result.size()*4,cudaMemcpyDeviceToHost));float expected=n*.125f*.125f*.75f;
 for(size_t i=0;i<result.size();i++){float reference=padded?(i%size_t(n)<17&&i/size_t(n)<23?19*.125f*.125f*.75f:0.f):expected;if(result[i]!=reference){std::cerr<<"incorrect value "<<result[i]<<", expected "<<reference<<'\n';return 4;}}
 if(argc>3&&n>=2048)for(int dimension:{128,256,2048,128}){
  cuda_bench::sm120_destroy(op);op=nullptr;
  check(cuda_bench::sm120_create(v,dimension,dimension,dimension,a,b,c,d,.75f,w,256ull*1024*1024,nullptr,&op));check(cuda_bench::sm120_launch(op,nullptr));check(cudaDeviceSynchronize());
  std::vector<float> values(size_t(dimension)*dimension);check(cudaMemcpy(values.data(),d,values.size()*4,cudaMemcpyDeviceToHost));
  for(auto x:values)if(x!=dimension*.125f*.125f*.75f){std::cerr<<"reused-descriptor result mismatch\n";return 4;}
 }
 cuda_bench::sm120_destroy(op);cudaFree(a);cudaFree(b);cudaFree(c);cudaFree(d);cudaFree(w);
 std::cout<<"known-answer passed: "<<cuda_bench::sm120_variant_name(v)<<", n="<<n<<", elements="<<result.size()<<", expected="<<expected<<"; no performance score\n";
 return 0;
}
