#include <cuda_runtime.h>
#include "reference.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <string>
static void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#define CUDA(x) checked(x)
struct Buffer {
    std::uint32_t*p=nullptr;
    explicit Buffer(std::size_t n){CUDA(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(*p)));}
    ~Buffer(){if(p)cudaFree(p);}
    Buffer(const Buffer&)=delete;Buffer&operator=(const Buffer&)=delete;
};
struct Stream {
    cudaStream_t s=nullptr;Stream(){CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}
    ~Stream(){if(s)cudaStreamDestroy(s);}
};
struct Event {
    cudaEvent_t e=nullptr;Event(){CUDA(cudaEventCreate(&e));}~Event(){if(e)cudaEventDestroy(e);}
};
__global__ void transform(const std::uint32_t*x,std::uint32_t*y,std::size_t n,unsigned rounds){
    const auto stride=static_cast<std::size_t>(blockDim.x)*gridDim.x;
    for(std::size_t i=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;i<n;i+=stride){
        auto value=x[i];for(unsigned k=0;k<rounds;++k)value=value*1664525u+1013904223u;y[i]=value;
    }
}
struct DeviceInfo {
    cudaDeviceProp props{};cudaFuncAttributes attributes{};int recommended=0,min_grid=0;
    DeviceInfo(){int device=0;CUDA(cudaGetDevice(&device));CUDA(cudaGetDeviceProperties(&props,device));
        CUDA(cudaFuncGetAttributes(&attributes,transform));
        CUDA(cudaOccupancyMaxPotentialBlockSize(&min_grid,&recommended,transform,0,0));
    }
    unsigned block(int policy)const{const unsigned fixed[]={128,256,512};return policy==3?static_cast<unsigned>(recommended):fixed[policy];}
    geometry_lab::Geometry launch(std::size_t n,unsigned b)const{
        return geometry_lab::geometry(n,b,static_cast<unsigned>(props.multiProcessorCount),
            static_cast<unsigned>(std::min(props.maxThreadsPerBlock,attributes.maxThreadsPerBlock)),static_cast<unsigned>(props.maxGridSize[0]));
    }
    int active(unsigned b)const{int blocks=0;CUDA(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,transform,static_cast<int>(b),0));return blocks;}
};
constexpr std::size_t guard=32;
constexpr std::uint32_t sentinel=0xdecafbad;
const char*names[]={"fixed128","fixed256","fixed512","occupancy"};
struct Work {
    std::size_t n;unsigned rounds;Buffer input,output;Stream stream;
    std::vector<std::uint32_t>original,expected;
    Work(std::size_t count,unsigned r):n(count),rounds(r),input(count+2*guard),output(count+2*guard),
        original(geometry_lab::input(count)),expected(geometry_lab::oracle(original,r)){
        std::vector<std::uint32_t>h(n+2*guard,sentinel);std::copy(original.begin(),original.end(),h.begin()+guard);
        CUDA(cudaMemcpyAsync(input.p,h.data(),h.size()*sizeof(h[0]),cudaMemcpyHostToDevice,stream.s));
        CUDA(cudaStreamSynchronize(stream.s));poison();
    }
    void poison(){std::vector<std::uint32_t>h(n+2*guard,sentinel);
        for(std::size_t i=0;i<n;++i)h[i+guard]=~expected[i];
        CUDA(cudaMemcpyAsync(output.p,h.data(),h.size()*sizeof(h[0]),cudaMemcpyHostToDevice,stream.s));CUDA(cudaStreamSynchronize(stream.s));
    }
    void run(geometry_lab::Geometry g){transform<<<g.grid,g.block,0,stream.s>>>(input.p+guard,output.p+guard,n,rounds);}
    void verify(){CUDA(cudaGetLastError());CUDA(cudaStreamSynchronize(stream.s));std::vector<std::uint32_t>h(n+2*guard);
        CUDA(cudaMemcpy(h.data(),output.p,h.size()*sizeof(h[0]),cudaMemcpyDeviceToHost));
        if(!std::equal(expected.begin(),expected.end(),h.begin()+guard))throw std::runtime_error("oracle mismatch");
        for(int buffer=0;buffer<2;++buffer){
            CUDA(cudaMemcpy(h.data(),buffer?output.p:input.p,h.size()*sizeof(h[0]),cudaMemcpyDeviceToHost));
            for(std::size_t i=0;i<guard;++i)if(h[i]!=sentinel||h[i+n+guard]!=sentinel)throw std::runtime_error("guard changed");
            if(!buffer&&!std::equal(original.begin(),original.end(),h.begin()+guard))throw std::runtime_error("input changed");
        }
    }
};
static void recovery(const DeviceInfo&info){
    Work w(257,1);CUDA(cudaGetLastError());
    w.run({static_cast<unsigned>(info.props.maxThreadsPerBlock+1),1});
    const auto error=cudaGetLastError();
    if(error!=cudaErrorInvalidConfiguration)throw std::runtime_error("invalid block was not rejected as expected");
    // The launch was rejected before execution. Do not introduce a memory fault.
    CUDA(cudaGetLastError());w.poison();w.run(info.launch(w.n,128));w.verify();
}
static void verify(bool quick,const DeviceInfo&info){
    std::size_t cases=0;
    for(std::size_t n:quick?std::vector<std::size_t>{1,257}:std::vector<std::size_t>{1,31,257,65537,1048576})
        for(unsigned rounds:{1u,64u}){Work w(n,rounds);
            for(int p=0;p<4;++p){w.poison();w.run(info.launch(n,info.block(p)));w.verify();++cases;}
        }
    if(!quick)recovery(info);
    std::cout<<"PASS cases="<<cases<<" exact_oracle=pass guards=pass input_immutable=pass invalid_launch_recovery="<<(quick?"not_run_quick":"pass")<<'\n';
}
static void diagnose(const DeviceInfo&info){int rt=0,dr=0;CUDA(cudaRuntimeGetVersion(&rt));CUDA(cudaDriverGetVersion(&dr));
    std::cout<<"gpu="<<info.props.name<<" cc="<<info.props.major<<'.'<<info.props.minor<<" runtime="<<rt<<" driver_api="<<dr
        <<" sm_count="<<info.props.multiProcessorCount<<" warp_size="<<info.props.warpSize<<" max_threads_per_sm="<<info.props.maxThreadsPerMultiProcessor
        <<" kernel_registers="<<info.attributes.numRegs<<" kernel_local_bytes="<<info.attributes.localSizeBytes
        <<" kernel_static_shared_bytes="<<info.attributes.sharedSizeBytes<<" occupancy_block="<<info.recommended
        <<" occupancy_min_grid="<<info.min_grid<<" seed=2027\n";
}
static void benchmark(const DeviceInfo&info){
    constexpr int warmup=5,repetitions=20,samples=9;
    std::cout<<"n,rounds,policy,block,grid,sm_count,max_threads_per_sm,active_blocks_per_sm,sample,repetitions,warmup,seed,device_ms,host_ms\n"<<std::setprecision(12);
    for(std::size_t n:{257u,65537u,1048576u})for(unsigned rounds:{1u,64u}){
        Work w(n,rounds);Event start,stop;
        for(int p=0;p<4;++p){auto g=info.launch(n,info.block(p));w.poison();for(int k=0;k<warmup;++k)w.run(g);w.verify();}
        for(int sample=0;sample<samples;++sample)for(int order=0;order<4;++order){int p=(sample+order)%4;
            auto g=info.launch(n,info.block(p));const int active=info.active(g.block);w.poison();
            auto t=std::chrono::steady_clock::now();CUDA(cudaEventRecord(start.e,w.stream.s));
            for(int k=0;k<repetitions;++k)w.run(g);
            CUDA(cudaGetLastError());CUDA(cudaEventRecord(stop.e,w.stream.s));CUDA(cudaEventSynchronize(stop.e));
            const double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/repetitions;
            float elapsed=0;CUDA(cudaEventElapsedTime(&elapsed,start.e,stop.e));w.verify();
            std::cout<<n<<','<<rounds<<','<<names[p]<<','<<g.block<<','<<g.grid<<','<<info.props.multiProcessorCount<<','<<info.props.maxThreadsPerMultiProcessor<<','
                <<active<<','<<sample<<','<<repetitions<<','<<warmup<<",2027,"<<elapsed/repetitions<<','<<host<<'\n';
        }
    }
}
int main(int argc,char**argv){try{
    if(argc!=2)throw std::invalid_argument("expected one mode");
    const std::string mode=argv[1];
    if(mode=="--help"){std::cout<<"geometry_lab --verify | --quick | --benchmark | --diagnose | --profile\n";return 0;}
    if(mode!="--verify"&&mode!="--quick"&&mode!="--benchmark"&&mode!="--diagnose"&&mode!="--profile")throw std::invalid_argument("unknown mode");
    DeviceInfo info;
    if(mode=="--verify"||mode=="--quick")verify(mode=="--quick",info);
    else if(mode=="--benchmark")benchmark(info);
    else if(mode=="--diagnose")diagnose(info);
    else{Work w(65537,64);w.run(info.launch(w.n,info.block(3)));w.verify();}
    return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
