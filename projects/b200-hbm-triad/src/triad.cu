#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void check(cudaError_t e) { if(e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
__global__ void scalar(const float* a, const float* b, float* c, size_t n) {
  for(size_t i=blockIdx.x*size_t(blockDim.x)+threadIdx.x; i<n; i+=size_t(gridDim.x)*blockDim.x)
    c[i]=a[i]+2.0f*b[i];
}
__global__ void vectorized(const float* a, const float* b, float* c, size_t n) {
  size_t t=blockIdx.x*size_t(blockDim.x)+threadIdx.x, step=size_t(gridDim.x)*blockDim.x;
  for(size_t i=t; i<n/4; i+=step) {
    float4 x=reinterpret_cast<const float4*>(a)[i], y=reinterpret_cast<const float4*>(b)[i];
    reinterpret_cast<float4*>(c)[i]=make_float4(x.x+2*y.x,x.y+2*y.y,x.z+2*y.z,x.w+2*y.w);
  }
  for(size_t i=(n/4)*4+t; i<n; i+=step) c[i]=a[i]+2*b[i];
}
// Independent deterministic input generation and scalar host oracle.
float input_a(size_t i) { return float(int(i%257)-128)*0.125f; }
float input_b(size_t i) { return float(int(i%127)-63)*0.25f; }
void launch(int mode, float* a,float* b,float* c,size_t n,int grid) {
  if(mode) vectorized<<<grid,256>>>(a,b,c,n); else scalar<<<grid,256>>>(a,b,c,n);
  check(cudaGetLastError());
}
int main(int argc,char** argv) { try {
  if(argc!=2 || (std::string(argv[1])!="verify" && std::string(argv[1])!="bench" && std::string(argv[1])!="profile"))
    throw std::runtime_error("usage: triad verify|bench|profile");
  cudaDeviceProp prop{}; check(cudaGetDeviceProperties(&prop,0));
  if(std::string(prop.name).find("B200")==std::string::npos || prop.major!=10)
    throw std::runtime_error("NVIDIA B200 required");
  std::cerr<<"gpu="<<prop.name<<" cc="<<prop.major<<'.'<<prop.minor<<" sms="<<prop.multiProcessorCount<<" l2_bytes="<<prop.l2CacheSize<<"\n";
  std::string task=argv[1];
  std::vector<size_t> sizes=task=="verify" ? std::vector<size_t>{1,3,4,5,31,257,4099,1048583} :
    task=="profile" ? std::vector<size_t>{67108864} : std::vector<size_t>{1048576,16777216,67108864,268435456};
  if(task=="bench") std::cout<<"n,variant,sample,warmup,repetitions,ms,logical_GBps\n";
  for(size_t n:sizes) {
    std::vector<float> a(n),b(n),out(n+4);
    for(size_t i=0;i<n;++i) { a[i]=input_a(i); b[i]=input_b(i); }
    float *da=nullptr,*db=nullptr,*dc=nullptr;
    check(cudaMalloc(&da,n*sizeof(float))); check(cudaMalloc(&db,n*sizeof(float))); check(cudaMalloc(&dc,(n+4)*sizeof(float)));
    check(cudaMemcpy(da,a.data(),n*4,cudaMemcpyHostToDevice)); check(cudaMemcpy(db,b.data(),n*4,cudaMemcpyHostToDevice));
    int grid=prop.multiProcessorCount*16;
    for(int mode=0;mode<2;++mode) {
      check(cudaMemset(dc,0x7f,(n+4)*4)); launch(mode,da,db,dc,n,grid); check(cudaDeviceSynchronize());
      check(cudaMemcpy(out.data(),dc,(n+4)*4,cudaMemcpyDeviceToHost));
      for(size_t i=0;i<n;++i) { double expected=double(input_a(i))+2.0*double(input_b(i));
        if(!std::isfinite(out[i]) || double(out[i])!=expected) throw std::runtime_error("oracle mismatch"); }
      for(size_t i=n;i<n+4;++i) { unsigned u=0; std::memcpy(&u,&out[i],4); if(u!=0x7f7f7f7fu) throw std::runtime_error("guard overwrite"); }
    }
    if(task=="bench") {
      cudaEvent_t start,stop; check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
      for(int m=0;m<2;++m) for(int w=0;w<5;++w) launch(m,da,db,dc,n,grid);
      check(cudaDeviceSynchronize());
      for(int s=0;s<9;++s) for(int order=0;order<2;++order) { int m=(s+order)%2;
        check(cudaEventRecord(start)); for(int r=0;r<10;++r) launch(m,da,db,dc,n,grid);
        check(cudaEventRecord(stop)); check(cudaEventSynchronize(stop)); float ms; check(cudaEventElapsedTime(&ms,start,stop)); ms/=10;
        std::cout<<n<<','<<(m?"float4":"scalar")<<','<<s<<",5,10,"<<ms<<','<<(double(n)*12/ms/1e6)<<'\n';
      }
      check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
    }
    check(cudaFree(da));check(cudaFree(db));check(cudaFree(dc));
  }
  std::cerr<<"PASS independent oracle, both variants, output guards\n";
  return 0;
} catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n';return 1; } }
