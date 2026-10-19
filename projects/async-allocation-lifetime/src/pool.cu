#include <cuda_runtime.h>
#include "reference.hpp"
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
static void checked(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#define CUDA(x) checked(x)
struct Stream {
    cudaStream_t s=nullptr;
    Stream(){CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}
    ~Stream(){if(s)cudaStreamDestroy(s);}
};
struct Event {
    cudaEvent_t e=nullptr;
    Event(){CUDA(cudaEventCreateWithFlags(&e,cudaEventDisableTiming));}
    ~Event(){if(e)cudaEventDestroy(e);}
};
struct Pool {
    cudaMemPool_t p=nullptr;
    explicit Pool(bool retain){int d=0;CUDA(cudaGetDevice(&d));cudaMemPoolProps props{};
        props.allocType=cudaMemAllocationTypePinned;props.handleTypes=cudaMemHandleTypeNone;
        props.location.type=cudaMemLocationTypeDevice;props.location.id=d;
        CUDA(cudaMemPoolCreate(&p,&props));
        std::uint64_t threshold=retain?UINT64_MAX:0;
        const auto result=cudaMemPoolSetAttribute(p,cudaMemPoolAttrReleaseThreshold,&threshold);
        if(result!=cudaSuccess){cudaMemPoolDestroy(p);p=nullptr;checked(result);}
    }
    ~Pool(){if(p)cudaMemPoolDestroy(p);}
};
struct Pinned {
    std::uint32_t*p=nullptr;
    explicit Pinned(std::size_t n){CUDA(cudaMallocHost(reinterpret_cast<void**>(&p),n*sizeof(*p)));}
    ~Pinned(){if(p)cudaFreeHost(p);}
};
__global__ void fill_pattern(std::uint32_t*p,std::size_t n){
    const std::size_t i=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;
    if(i<n)p[i]=static_cast<std::uint32_t>(i)*2654435761u+2027u;
}
constexpr std::size_t guard=32;
constexpr std::uint32_t sentinel=0xaabbccddu;
const char*names[]={"legacy","async_release","async_retain"};
struct Allocator {
    int path;Pool pool;Stream producer,consumer;Event ready,done;
    explicit Allocator(int p):path(p),pool(p==2){}
    std::uint32_t* allocate(std::size_t bytes){std::uint32_t*p=nullptr;
        if(path==0)CUDA(cudaMalloc(reinterpret_cast<void**>(&p),bytes));
        else CUDA(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&p),bytes,pool.p,producer.s));
        return p;
    }
    void release(std::uint32_t*p){
        if(path==0){CUDA(cudaStreamSynchronize(producer.s));CUDA(cudaFree(p));}
        else{CUDA(cudaFreeAsync(p,producer.s));CUDA(cudaStreamSynchronize(producer.s));}
    }
    void iteration(std::size_t n){auto*p=allocate((n+2*guard)*sizeof(std::uint32_t));
        fill_pattern<<<static_cast<unsigned>((n+255)/256),256,0,producer.s>>>(p+guard,n);
        CUDA(cudaGetLastError());release(p);
    }
    void correctness(std::size_t n,bool cross_stream){
        auto expected=pool_lab::oracle(n);Pinned output(n+2*guard);
        std::fill(output.p,output.p+n+2*guard,sentinel);
        for(std::size_t i=0;i<n;++i)output.p[i+guard]=~expected[i];
        auto*p=allocate((n+2*guard)*sizeof(std::uint32_t));
        // Queue initialization on the allocating stream, never the default stream.
        CUDA(cudaMemcpyAsync(p,output.p,(n+2*guard)*sizeof(*p),cudaMemcpyHostToDevice,producer.s));
        cudaStream_t work=producer.s;
        if(cross_stream){
            if(path==0)throw std::invalid_argument("cross-stream mode requires async path");
            CUDA(cudaEventRecord(ready.e,producer.s));CUDA(cudaStreamWaitEvent(consumer.s,ready.e,0));work=consumer.s;
        }
        fill_pattern<<<static_cast<unsigned>((n+255)/256),256,0,work>>>(p+guard,n);
        CUDA(cudaGetLastError());
        CUDA(cudaMemcpyAsync(output.p,p,(n+2*guard)*sizeof(*p),cudaMemcpyDeviceToHost,work));
        if(cross_stream){CUDA(cudaEventRecord(done.e,consumer.s));CUDA(cudaStreamWaitEvent(producer.s,done.e,0));}
        release(p);
        if(!std::equal(expected.begin(),expected.end(),output.p+guard))throw std::runtime_error("oracle mismatch");
        for(std::size_t i=0;i<guard;++i)if(output.p[i]!=sentinel||output.p[i+n+guard]!=sentinel)throw std::runtime_error("guard changed");
        verify_unused();
    }
    std::uint64_t verify_unused(){
        std::uint64_t used=0;
        if(path){CUDA(cudaMemPoolGetAttribute(pool.p,cudaMemPoolAttrUsedMemCurrent,&used));
            if(used)throw std::runtime_error("pool retains live allocation after synchronized free");}
        return used;
    }
    std::uint64_t reserved(){std::uint64_t bytes=0;if(path)CUDA(cudaMemPoolGetAttribute(pool.p,cudaMemPoolAttrReservedMemCurrent,&bytes));return bytes;}
};
static void require_support(){int d=0,supported=0;CUDA(cudaGetDevice(&d));
    CUDA(cudaDeviceGetAttribute(&supported,cudaDevAttrMemoryPoolsSupported,d));
    if(!supported)throw std::runtime_error("device does not support stream-ordered memory pools");
}
static void verify(bool quick){
    std::size_t cases=0;
    for(std::size_t n:quick?std::vector<std::size_t>{1,257}:std::vector<std::size_t>{1,31,256,257,65537,1048576})
        for(int p=0;p<3;++p){Allocator a(p);
            for(int repeat=0;repeat<3;++repeat){a.correctness(n,false);}
            ++cases;
            if(p){for(int repeat=0;repeat<3;++repeat)a.correctness(n,true);++cases;}
        }
    std::cout<<"PASS cases="<<cases<<" repeats=3 exact_oracle=pass guards=pass event_handoff=pass zero_live_pool_bytes=pass\n";
}
static void benchmark(){
    constexpr int warmup=3,repetitions=20,samples=9;
    std::cout<<"n,path,sample,repetitions,warmup,seed,host_ms,reserved_bytes,used_bytes\n"<<std::setprecision(12);
    for(std::size_t n:{256u,65537u,1048576u}){
        Allocator legacy(0),release(1),retain(2);Allocator*paths[]={&legacy,&release,&retain};
        for(auto*a:paths){a->correctness(n,false);for(int k=0;k<warmup;++k)a->iteration(n);a->verify_unused();}
        for(int sample=0;sample<samples;++sample)for(int order=0;order<3;++order){int p=(sample+order)%3;auto*a=paths[p];
            auto t=std::chrono::steady_clock::now();for(int k=0;k<repetitions;++k)a->iteration(n);
            const double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/repetitions;
            const auto used=a->verify_unused();const auto reserved=a->reserved();
            // Verify an observable result after every measured sample, outside timing.
            a->correctness(n,false);
            std::cout<<n<<','<<names[p]<<','<<sample<<','<<repetitions<<','<<warmup<<",2027,"<<elapsed<<','<<reserved<<','<<used<<'\n';
        }
    }
}
int main(int argc,char**argv){try{
    if(argc!=2)throw std::invalid_argument("expected one mode");
    std::string mode=argv[1];
    if(mode=="--help"){std::cout<<"pool_lab --verify | --quick | --benchmark | --diagnose | --profile\n";return 0;}
    if(mode!="--verify"&&mode!="--quick"&&mode!="--benchmark"&&mode!="--diagnose"&&mode!="--profile")throw std::invalid_argument("unknown mode");
    require_support();
    if(mode=="--verify"||mode=="--quick")verify(mode=="--quick");
    else if(mode=="--benchmark")benchmark();
    else if(mode=="--profile"){Allocator a(2);a.correctness(65537,true);a.iteration(65537);a.verify_unused();}
    else{int d=0,rt=0,dr=0;cudaDeviceProp props{};CUDA(cudaGetDevice(&d));CUDA(cudaGetDeviceProperties(&props,d));
        CUDA(cudaRuntimeGetVersion(&rt));CUDA(cudaDriverGetVersion(&dr));
        std::cout<<"gpu="<<props.name<<" cc="<<props.major<<'.'<<props.minor<<" runtime="<<rt<<" driver_api="<<dr<<" memory_pools_supported=1 seed=2027 block=256\n";
    }return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
