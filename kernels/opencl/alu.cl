// Bounded register recurrence. Every accumulator and lane is written once.
// multiplier/addends are runtime arguments; no volatile/global hot-loop access.
#if HALF_MODE
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#if LANES == 2
typedef half2 value_t;
#else
typedef half value_t;
#endif
#else
typedef float value_t;
#endif
__kernel void alu(__global float* out, uint iterations, uint seed, float multiplier, float step) {
    const uint gid=get_global_id(0);
    value_t x[ACC],b[ACC];
    #pragma unroll
    for(uint a=0;a<ACC;a++) {
        float v=(float)((gid*17+a*13+seed)&255)/512.0f+0.125f;
        float s=((gid+a+seed)&1)?-1.0f:1.0f;
        float add=(float)(1+a%3)*step*s;
        #if HALF_MODE && LANES == 2
        float v1=(float)((gid*17+a*13+7+seed)&255)/512.0f+0.125f;
        float s1=((gid+a+1+seed)&1)?-1.0f:1.0f;
        x[a]=(half2)((half)(v*s),(half)(v1*s1));
        b[a]=(half2)((half)add,(half)((float)(1+(a+1)%3)*step*s1));
        #else
        x[a]=(value_t)(v*s);b[a]=(value_t)add;
        #endif
    }
    value_t m=(value_t)multiplier;
    for(uint i=0;i<iterations;i++) {
        #pragma unroll
        for(uint a=0;a<ACC;a++) x[a]=fma(x[a],m,b[a]);
    }
    #pragma unroll
    for(uint a=0;a<ACC;a++) {
        #if HALF_MODE && LANES == 2
        vstore2(convert_float2(x[a]),0,out+((size_t)gid*ACC+a)*2);
        #else
        out[(size_t)gid*ACC+a]=(float)x[a];
        #endif
    }
}
