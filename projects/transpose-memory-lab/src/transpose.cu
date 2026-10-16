#include <cuda_runtime.h>
#include "reference.hpp"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

static void check(cudaError_t e, const char* call) {
    if (e != cudaSuccess)
        throw std::runtime_error(std::string(call)+": "+cudaGetErrorString(e));
}
#define CUDA(call) check((call), #call)

constexpr int TILE=32, BLOCK_ROWS=8, GUARD=64;
constexpr float SENTINEL=-1234567.0f;

__global__ void copy_kernel(const float* in,float* out,int rows,int cols) {
    size_t n=size_t(rows)*cols;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x; i<n;
        i+=size_t(blockDim.x)*gridDim.x) out[i]=in[i];
}
__global__ void naive_kernel(const float* in,float* out,int rows,int cols) {
    int x=blockIdx.x*TILE+threadIdx.x, y=blockIdx.y*TILE+threadIdx.y;
    for(int j=0;j<TILE;j+=BLOCK_ROWS)
        if(x<cols && y+j<rows) out[size_t(x)*rows+y+j]=in[size_t(y+j)*cols+x];
}
template<int PAD>
__global__ void tiled_kernel(const float* in,float* out,int rows,int cols) {
    __shared__ float tile[TILE][TILE+PAD];
    int x=blockIdx.x*TILE+threadIdx.x, y=blockIdx.y*TILE+threadIdx.y;
    for(int j=0;j<TILE;j+=BLOCK_ROWS)
        if(x<cols && y+j<rows) tile[threadIdx.y+j][threadIdx.x]=in[size_t(y+j)*cols+x];
    // All threads, including edge threads, must reach the barrier.
    __syncthreads();
    x=blockIdx.y*TILE+threadIdx.x;
    y=blockIdx.x*TILE+threadIdx.y;
    for(int j=0;j<TILE;j+=BLOCK_ROWS)
        if(x<rows && y+j<cols) out[size_t(y+j)*rows+x]=tile[threadIdx.x][threadIdx.y+j];
}
enum Variant { COPY, NAIVE, TILED, PADDED };
static const char* names[]={"copy","naive","tiled","padded"};
static void launch(Variant v,const float* in,float* out,int r,int c) {
    dim3 block(TILE,BLOCK_ROWS), grid((c+TILE-1)/TILE,(r+TILE-1)/TILE);
    switch(v) {
        case COPY: copy_kernel<<<std::min<size_t>((size_t(r)*c+255)/256,65535),256>>>(in,out,r,c); break;
        case NAIVE: naive_kernel<<<grid,block>>>(in,out,r,c); break;
        case TILED: tiled_kernel<0><<<grid,block>>>(in,out,r,c); break;
        case PADDED: tiled_kernel<1><<<grid,block>>>(in,out,r,c); break;
    }
    CUDA(cudaGetLastError());
}
struct Buffer {
    float* p=nullptr;
    explicit Buffer(size_t n) { CUDA(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(float))); }
    ~Buffer() { if(p) cudaFree(p); }
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
};
struct Event {
    cudaEvent_t e;
    Event(){CUDA(cudaEventCreate(&e));}
    ~Event(){cudaEventDestroy(e);}
    Event(const Event&)=delete;
};
static void verify(int r,int c) {
    size_t n=size_t(r)*c;
    std::vector<float> input(n), actual(n+2*GUARD), guarded(n+2*GUARD,SENTINEL);
    std::mt19937 rng(2027);
    for(auto& x:input) x=float(int(rng()%200001)-100000)/128.0f;
    auto expected=transpose_reference(input,r,c);
    Buffer in(n+2*GUARD),out(n+2*GUARD);
    std::copy(input.begin(),input.end(),guarded.begin()+GUARD);
    CUDA(cudaMemcpy(in.p,guarded.data(),guarded.size()*sizeof(float),cudaMemcpyHostToDevice));
    for(int v=COPY;v<=PADDED;++v) {
        std::fill(actual.begin(),actual.end(),SENTINEL);
        // Poison the interior so a skipped write cannot silently pass.
        std::fill(actual.begin()+GUARD,actual.begin()+GUARD+n,
                  std::numeric_limits<float>::quiet_NaN());
        CUDA(cudaMemcpy(out.p,actual.data(),actual.size()*sizeof(float),cudaMemcpyHostToDevice));
        launch(Variant(v),in.p+GUARD,out.p+GUARD,r,c);
        CUDA(cudaDeviceSynchronize());
        CUDA(cudaMemcpy(actual.data(),out.p,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for(size_t i=0;i<n;++i)
            if(actual[i+GUARD] != (v==COPY?input[i]:expected[i]))
                throw std::runtime_error(std::string(names[v])+" mismatch at "+std::to_string(i));
        for(int i=0;i<GUARD;++i)
            if(actual[i]!=SENTINEL || actual[n+GUARD+i]!=SENTINEL)
                throw std::runtime_error("output guard overwritten");
    }
    CUDA(cudaMemcpy(actual.data(),in.p,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
    if(actual!=guarded) throw std::runtime_error("input modified");
    std::cerr<<"PASS shape="<<r<<"x"<<c<<" variants=4 guards=pass\n";
}
static void benchmark(int r,int c,int repetitions) {
    size_t n=size_t(r)*c;
    Buffer in(n),out(n);
    CUDA(cudaMemset(in.p,0,n*sizeof(float)));
    Event start,stop;
    std::cout<<"variant,rows,cols,sample,iterations,total_ms,ms_per_launch,effective_GBps\n";
    // Keep tiny shapes long enough to reduce CUDA-event resolution effects.
    int iterations=n<1024*1024?100:10;
    for(int v=COPY;v<=PADDED;++v) {
        for(int i=0;i<10;++i) launch(Variant(v),in.p,out.p,r,c);
        CUDA(cudaDeviceSynchronize());
        for(int sample=0;sample<repetitions;++sample) {
            CUDA(cudaEventRecord(start.e));
            for(int i=0;i<iterations;++i) launch(Variant(v),in.p,out.p,r,c);
            CUDA(cudaEventRecord(stop.e)); CUDA(cudaEventSynchronize(stop.e));
            float elapsed=0; CUDA(cudaEventElapsedTime(&elapsed,start.e,stop.e));
            if(elapsed<=0) throw std::runtime_error("nonpositive event duration");
            double ms=double(elapsed)/iterations;
            std::cout<<names[v]<<','<<r<<','<<c<<','<<sample<<','<<iterations<<','
                     <<elapsed<<','<<ms<<','<<(2.0*n*sizeof(float)/(ms*1e6))<<'\n';
        }
    }
}
static int positive(const char* s,int maximum) {
    size_t consumed=0;
    long long n=std::stoll(s,&consumed);
    if(consumed!=std::string(s).size() || n<1 || n>maximum)
        throw std::invalid_argument("argument outside supported positive range");
    return int(n);
}
int main(int argc,char** argv) {
    try {
        std::string mode=argc>1?argv[1]:"--help";
        if(mode=="--help") {
            std::cout<<"transpose_lab --verify | --diagnose | --benchmark ROWS COLS SAMPLES\n";
            return 0;
        }
        if(mode!="--verify" && mode!="--diagnose" && mode!="--benchmark")
            throw std::invalid_argument("unknown mode; use --help");
        int r=0,c=0,samples=0;
        if(mode=="--benchmark") {
            if(argc!=5) throw std::invalid_argument("benchmark needs ROWS COLS SAMPLES");
            r=positive(argv[2],32768); c=positive(argv[3],32768); samples=positive(argv[4],1000);
            if(size_t(r)*c>(size_t(1)<<28)) throw std::invalid_argument("matrix exceeds 2^28 elements");
        } else if(argc!=2) throw std::invalid_argument("unexpected extra arguments");
        int count=0;
        cudaError_t status=cudaGetDeviceCount(&count);
        if(status!=cudaSuccess || count==0) {
            std::cerr<<"GPU_UNAVAILABLE: "<<cudaGetErrorString(status)<<"; devices="<<count
                     <<". Run inside a GPU allocation.\n";
            return 77;
        }
        cudaDeviceProp prop; CUDA(cudaGetDeviceProperties(&prop,0));
        int runtime=0,driver=0; CUDA(cudaRuntimeGetVersion(&runtime)); CUDA(cudaDriverGetVersion(&driver));
        std::cerr<<"device="<<prop.name<<" cc="<<prop.major<<'.'<<prop.minor
                 <<" runtime="<<runtime<<" driver_api="<<driver<<" seed=2027\n";
        std::cout<<std::setprecision(10);
        if(mode=="--verify") {
            for(auto shape:std::vector<std::pair<int,int>>{{1,1},{1,65},{65,1},{31,33},{32,32},
                            {33,31},{65,97},{257,513},{1024,1024},{1025,2049}})
                verify(shape.first,shape.second);
        } else if(mode=="--benchmark") {
            verify(r,c); benchmark(r,c,samples);
        }
        return 0;
    } catch(const std::exception& e) { std::cerr<<"ERROR: "<<e.what()<<'\n'; return 1; }
}
