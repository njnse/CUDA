#include <cuda_runtime.h>
#include <cub/device/device_reduce.cuh>
#include "reference.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

static void checked(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
#define CUDA(x) checked(x)
struct Buffer {
    void* p=nullptr;
    explicit Buffer(std::size_t bytes) { CUDA(cudaMalloc(&p,bytes)); }
    ~Buffer(){if(p)cudaFree(p);}
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    template<class T> T* get(){return static_cast<T*>(p);}
};
struct Stream {
    cudaStream_t s=nullptr;
    Stream(){CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}
    ~Stream(){if(s)cudaStreamDestroy(s);}
};
struct Event {
    cudaEvent_t e=nullptr;
    Event(){CUDA(cudaEventCreate(&e));}
    ~Event(){if(e)cudaEventDestroy(e);}
};
template<class In,class Acc>
__global__ void reduce_tree(const In* x,Acc* y,std::size_t n) {
    __shared__ Acc tile[256];
    Acc value=0;
    for(std::size_t i=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;
        i<n;i+=static_cast<std::size_t>(gridDim.x)*blockDim.x) value+=static_cast<Acc>(x[i]);
    tile[threadIdx.x]=value;
    __syncthreads();
    for(unsigned stride=128;stride;stride/=2){
        if(threadIdx.x<stride)tile[threadIdx.x]+=tile[threadIdx.x+stride];
        __syncthreads();
    }
    if(!threadIdx.x)y[blockIdx.x]=tile[0];
}
constexpr std::size_t guard=32;
constexpr float sentinel=12345.5f;
struct Work {
    std::size_t n,blocks,temp_bytes=0;
    Buffer input,partial32,partial64,out32,out64;
    Stream stream;
    Buffer* scratch=nullptr;
    std::vector<float> host;
    reduction_lab::Reference reference;
    bool exact_fp32;
    Work(std::size_t count,const std::string& dataset):n(count),blocks(std::min<std::size_t>((count+255)/256,1024)),
        input((count+2*guard)*sizeof(float)),partial32(blocks*sizeof(float)),partial64(blocks*sizeof(double)),
        out32((1+2*guard)*sizeof(float)),out64((1+2*guard)*sizeof(double)),host(reduction_lab::input(count,dataset)),reference(reduction_lab::oracle(host)),exact_fp32(dataset=="positive") {
        std::vector<float> h(n+2*guard,sentinel);
        std::copy(host.begin(),host.end(),h.begin()+guard);
        CUDA(cudaMemcpyAsync(input.p,h.data(),h.size()*sizeof(float),cudaMemcpyHostToDevice,stream.s));
        CUDA(cudaStreamSynchronize(stream.s));
        CUDA(cub::DeviceReduce::Sum(nullptr,temp_bytes,input.get<float>()+guard,out32.get<float>()+guard,static_cast<int>(n),stream.s));
        scratch=new Buffer(temp_bytes?temp_bytes:1);
    }
    ~Work(){delete scratch;}
    void poison() {
        std::vector<float> f(1+2*guard,sentinel); f[guard]=NAN;
        std::vector<double> d(1+2*guard,sentinel); d[guard]=NAN;
        CUDA(cudaMemcpyAsync(out32.p,f.data(),f.size()*sizeof(float),cudaMemcpyHostToDevice,stream.s));
        CUDA(cudaMemcpyAsync(out64.p,d.data(),d.size()*sizeof(double),cudaMemcpyHostToDevice,stream.s));
        CUDA(cudaStreamSynchronize(stream.s));
    }
    void launch(int path) {
        if(path==0) {
            reduce_tree<float,float><<<static_cast<unsigned>(blocks),256,0,stream.s>>>(input.get<float>()+guard,partial32.get<float>(),n);
            reduce_tree<float,float><<<1,256,0,stream.s>>>(partial32.get<float>(),out32.get<float>()+guard,blocks);
        } else if(path==1) {
            reduce_tree<float,double><<<static_cast<unsigned>(blocks),256,0,stream.s>>>(input.get<float>()+guard,partial64.get<double>(),n);
            reduce_tree<double,double><<<1,256,0,stream.s>>>(partial64.get<double>(),out64.get<double>()+guard,blocks);
        } else CUDA(cub::DeviceReduce::Sum(scratch->p,temp_bytes,input.get<float>()+guard,out32.get<float>()+guard,static_cast<int>(n),stream.s));
    }
    long double verify(int path) {
        CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(stream.s));
        double value=0;
        if(path==1){
            std::vector<double> y(1+2*guard);
            CUDA(cudaMemcpy(y.data(),out64.p,y.size()*sizeof(double),cudaMemcpyDeviceToHost));
            value=y[guard];
            for(std::size_t i=0;i<guard;++i)if(y[i]!=sentinel||y[i+guard+1]!=sentinel)throw std::runtime_error("output guard changed");
        }else{
            std::vector<float> y(1+2*guard);
            CUDA(cudaMemcpy(y.data(),out32.p,y.size()*sizeof(float),cudaMemcpyDeviceToHost));
            value=y[guard];
            for(std::size_t i=0;i<guard;++i)if(y[i]!=sentinel||y[i+guard+1]!=sentinel)throw std::runtime_error("output guard changed");
        }
        if(!std::isfinite(value))throw std::runtime_error("nonfinite result");
        const long double error=std::fabs(static_cast<long double>(value)-reference.sum);
        // All chosen inputs have exact sums in FP64. FP64 must match exactly.
        // FP32 uses an explicitly conservative summation bound, not a fixed tolerance.
        const long double bound=(path==1 || exact_fp32) ? 0 : reduction_lab::gamma(n,0x1p-24L)*reference.sum_abs;
        if(error>bound)throw std::runtime_error("sum exceeded numerical contract");
        std::vector<float> x(n+2*guard);
        CUDA(cudaMemcpy(x.data(),input.p,x.size()*sizeof(float),cudaMemcpyDeviceToHost));
        if(!std::equal(host.begin(),host.end(),x.begin()+guard))throw std::runtime_error("input changed");
        for(std::size_t i=0;i<guard;++i)if(x[i]!=sentinel||x[i+n+guard]!=sentinel)throw std::runtime_error("input guard changed");
        return error;
    }
};
const char* names[]={"tree_fp32","tree_fp64","cub_fp32"};
static void verify(bool quick){
    std::size_t cases=0;
    for(std::size_t n: quick?std::vector<std::size_t>{1,257}:std::vector<std::size_t>{1,31,255,256,257,65537,1048576})
        for(const auto& dataset:{"positive","cancellation","dyadic"}) {
            Work w(n,dataset);
            for(int p=0;p<3;++p){w.poison();w.launch(p);auto e=w.verify(p);
                std::cout<<"PASS n="<<n<<" dataset="<<dataset<<" path="<<names[p]<<" abs_error="<<e<<'\n';++cases;}
        }
    std::cout<<"PASS cases="<<cases<<" oracle=pass poison=pass guards=pass input_immutable=pass\n";
}
static void benchmark(){
    constexpr int warmup=5,repetitions=30,samples=9;
    std::cout<<"n,dataset,path,sample,repetitions,warmup,seed,device_ms,host_ms,abs_error,sum_abs,reference_sum\n"<<std::setprecision(18);
    for(std::size_t n:{257u,65537u,1048576u})for(const auto& dataset:{"positive","cancellation","dyadic"}){
        Work w(n,dataset);Event start,stop;
        for(int p=0;p<3;++p){w.poison();for(int k=0;k<warmup;++k)w.launch(p);w.verify(p);}
        for(int sample=0;sample<samples;++sample)for(int order=0;order<3;++order){
            int p=(sample+order)%3;
            CUDA(cudaStreamSynchronize(w.stream.s));
            auto t=std::chrono::steady_clock::now();CUDA(cudaEventRecord(start.e,w.stream.s));
            for(int k=0;k<repetitions;++k)w.launch(p);
            CUDA(cudaGetLastError());CUDA(cudaEventRecord(stop.e,w.stream.s));CUDA(cudaEventSynchronize(stop.e));
            double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/repetitions;
            float elapsed=0;CUDA(cudaEventElapsedTime(&elapsed,start.e,stop.e));auto error=w.verify(p);
            std::cout<<n<<','<<dataset<<','<<names[p]<<','<<sample<<','<<repetitions<<','<<warmup<<",2027,"<<elapsed/repetitions<<','<<host<<','<<error<<','<<w.reference.sum_abs<<','<<w.reference.sum<<'\n';
        }
    }
}
int main(int argc,char** argv){
    try{
        if(argc!=2)throw std::invalid_argument("expected one mode");
        std::string mode=argv[1];
        if(mode=="--help")std::cout<<"reduction_lab --verify | --quick | --benchmark | --diagnose | --profile\n";
        else if(mode=="--verify"||mode=="--quick")verify(mode=="--quick");
        else if(mode=="--benchmark")benchmark();
        else if(mode=="--profile"){Work w(65537,"cancellation");for(int p=0;p<3;++p){w.poison();w.launch(p);w.verify(p);}}
        else if(mode=="--diagnose"){
            int device=0,rt=0,dr=0;cudaDeviceProp props{};CUDA(cudaGetDevice(&device));CUDA(cudaGetDeviceProperties(&props,device));
            CUDA(cudaRuntimeGetVersion(&rt));CUDA(cudaDriverGetVersion(&dr));
            std::cout<<"gpu="<<props.name<<" cc="<<props.major<<'.'<<props.minor<<" runtime="<<rt<<" driver_api="<<dr<<" block=256 seed=2027\n";
        }else throw std::invalid_argument("unknown mode");
        return 0;
    }catch(const std::exception& e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}
}
