// Experimental exhaustive configuration search using the locked cuBLASLt API.
// Does not change compute type, scale, layouts, fast accumulation or validation.
json extended_search(GemmPlan& p,GemmBuffers& b,cudaStream_t stream,json seed){
 auto ids=reinterpret_cast<decltype(&::cublasLtMatmulAlgoGetIds)>(GetProcAddress(p.lib.module,"cublasLtMatmulAlgoGetIds"));
 auto init=reinterpret_cast<decltype(&::cublasLtMatmulAlgoInit)>(GetProcAddress(p.lib.module,"cublasLtMatmulAlgoInit"));
 auto cap=reinterpret_cast<decltype(&::cublasLtMatmulAlgoCapGetAttribute)>(GetProcAddress(p.lib.module,"cublasLtMatmulAlgoCapGetAttribute"));
 auto set=reinterpret_cast<decltype(&::cublasLtMatmulAlgoConfigSetAttribute)>(GetProcAddress(p.lib.module,"cublasLtMatmulAlgoConfigSetAttribute"));
 if(!ids||!init||!cap||!set)throw std::runtime_error("extended search symbols missing");
 const auto type=p.dtype=="fp16"?CUDA_R_16F:CUDA_R_8F_E4M3;
 std::array<int,256> available{};int count=0;
 blas_check(ids(p.lib.handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,type,type,CUDA_R_32F,CUDA_R_32F,int(available.size()),available.data(),&count),"enumerate IDs");
 json audit={{"algorithm_ids",json::array()},{"checked",0},{"timed",json::array()},{"seed",seed}};
 auto winner=p.algorithm;Timer timer(stream);
 auto measure=[&](){return timer.time(3,[&](uint64_t){p.execute(b,stream);}).device_ms/3;};
 for(int i=0;i<5;i++)p.execute(b,stream);cuda_check(cudaStreamSynchronize(stream),"extended warmup");
 double best=measure();audit["seed_ms"]=best;
 const auto start=std::chrono::steady_clock::now();bool timeout=false;
 for(int i=0;i<count&&!timeout;i++){
  const int id=available[i];audit["algorithm_ids"].push_back(id);cublasLtMatmulAlgo_t base{};
  if(init(p.lib.handle,CUBLAS_COMPUTE_32F,CUDA_R_32F,type,type,CUDA_R_32F,CUDA_R_32F,id,&base)!=CUBLAS_STATUS_SUCCESS)continue;
  auto options=[&](cublasLtMatmulAlgoCapAttributes_t attr){size_t bytes=0;if(cap(&base,attr,nullptr,0,&bytes)!=CUBLAS_STATUS_SUCCESS||!bytes)return std::vector<int>{0};std::vector<int> v(bytes/sizeof(int));if(cap(&base,attr,v.data(),bytes,&bytes)!=CUBLAS_STATUS_SUCCESS)return std::vector<int>{0};return v;};
  const auto tiles=options(CUBLASLT_ALGO_CAP_TILE_IDS),stages=options(CUBLASLT_ALGO_CAP_STAGES_IDS);
  int custommax=0,swizzlemax=0;size_t written=0;cap(&base,CUBLASLT_ALGO_CAP_CUSTOM_OPTION_MAX,&custommax,sizeof(custommax),&written);cap(&base,CUBLASLT_ALGO_CAP_CTA_SWIZZLING_SUPPORT,&swizzlemax,sizeof(swizzlemax),&written);
  for(int tile:tiles)for(int stage:stages)for(int custom=0;custom<=custommax;custom++)for(int swizzle=0;swizzle<=std::min(swizzlemax,1);swizzle++){
   if(timeout)continue;
   if(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>45){timeout=true;continue;}
   auto algo=base;
   if(set(&algo,CUBLASLT_ALGO_CONFIG_TILE_ID,&tile,sizeof(tile))!=CUBLAS_STATUS_SUCCESS||set(&algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,&stage,sizeof(stage))!=CUBLAS_STATUS_SUCCESS||set(&algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,&custom,sizeof(custom))!=CUBLAS_STATUS_SUCCESS||set(&algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,&swizzle,sizeof(swizzle))!=CUBLAS_STATUS_SUCCESS)continue;
   audit["checked"]=audit["checked"].get<int>()+1;
   cublasLtMatmulHeuristicResult_t checked{};if(p.lib.cublasLtMatmulAlgoCheck(p.lib.handle,p.op,p.a,p.b,p.c,p.d,&algo,&checked)!=CUBLAS_STATUS_SUCCESS||checked.state!=CUBLAS_STATUS_SUCCESS||checked.workspaceSize>p.workspace.bytes)continue;
   p.algorithm=algo;p.execute(b,stream);cuda_check(cudaStreamSynchronize(stream),"extended initialize");
   const auto first=timer.time(1,[&](uint64_t){p.execute(b,stream);}).device_ms;
   if(first>100)continue;
   std::vector<double> samples{measure(),measure(),measure()};double ms=gb::median(samples);
   audit["timed"].push_back({{"id",id},{"tile",tile},{"stage",stage},{"custom",custom},{"swizzle",swizzle},{"sample_ms",samples},{"median_ms",ms},{"workspace",checked.workspaceSize}});
   if(ms<best){best=ms;winner=algo;audit["selected"]=audit["timed"].back();}
  }
 }
 p.algorithm=winner;audit["time_budget_exhausted"]=timeout;audit["best_ms"]=best;return audit;
}
