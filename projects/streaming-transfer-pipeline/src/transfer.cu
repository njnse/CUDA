#include <cuda_runtime.h>
#include "reference.hpp"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
static void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
#define CUDA(x) check(x)
struct Device {
    std::uint32_t*p=nullptr;
    explicit Device(std::size_t n){CUDA(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(*p)));}
    ~Device(){if(p)cudaFree(p);}
    Device(const Device&)=delete;Device&operator=(const Device&)=delete;
};
struct Pinned {
    std::uint32_t*p=nullptr;
    explicit Pinned(std::size_t n){CUDA(cudaMallocHost(reinterpret_cast<void**>(&p),n*sizeof(*p)));}
    ~Pinned(){if(p)cudaFreeHost(p);}
    Pinned(const Pinned&)=delete;Pinned&operator=(const Pinned&)=delete;
};
struct Stream {
    cudaStream_t s=nullptr;
    Stream(){CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking));}
    ~Stream(){if(s)cudaStreamDestroy(s);}
};
__global__ void affine(const std::uint32_t*x,std::uint32_t*y,std::size_t n){
    const std::size_t i=blockIdx.x*static_cast<std::size_t>(blockDim.x)+threadIdx.x;
    if(i<n)y[i]=x[i]*1664525u+1013904223u;
}
constexpr std::size_t guard=32;
constexpr std::uint32_t sentinel=0xdecafbad;
const char*names[]={"pageable_serial","pinned_serial","pinned_pipeline"};
struct Work {
    std::size_t n,chunk;
    std::vector<std::uint32_t>original,expected,page_in,page_out;
    Pinned pin_in,pin_out;
    std::unique_ptr<Device>din[2],dout[2];
    Stream streams[2];
    std::vector<transfer_lab::Chunk>serial,parallel;
    std::size_t last_offset[2]={0,0},last_count[2]={0,0};
    Work(std::size_t count,std::size_t c):n(count),chunk(c),original(transfer_lab::input(count)),expected(transfer_lab::oracle(original)),
        page_in(count+2*guard,sentinel),page_out(count+2*guard,sentinel),pin_in(count+2*guard),pin_out(count+2*guard),
        serial(transfer_lab::plan(count,c,1)),parallel(transfer_lab::plan(count,c,2)){
        std::copy(original.begin(),original.end(),page_in.begin()+guard);
        std::fill(pin_in.p,pin_in.p+n+2*guard,sentinel);
        std::copy(original.begin(),original.end(),pin_in.p+guard);
        for(int s=0;s<2;++s){
            din[s]=std::make_unique<Device>(chunk+2*guard);dout[s]=std::make_unique<Device>(chunk+2*guard);
            std::vector<std::uint32_t>h(chunk+2*guard,sentinel);
            CUDA(cudaMemcpyAsync(din[s]->p,h.data(),h.size()*sizeof(h[0]),cudaMemcpyHostToDevice,streams[s].s));
            CUDA(cudaMemcpyAsync(dout[s]->p,h.data(),h.size()*sizeof(h[0]),cudaMemcpyHostToDevice,streams[s].s));
            CUDA(cudaStreamSynchronize(streams[s].s));
        }
        poison();
    }
    void poison(){
        std::fill(page_out.begin(),page_out.end(),sentinel);
        std::fill(pin_out.p,pin_out.p+n+2*guard,sentinel);
        // Ensure every expected element differs from its initial output value.
        for(std::size_t i=0;i<n;++i){page_out[i+guard]=~expected[i];pin_out.p[i+guard]=~expected[i];}
    }
    void run(int path){
        const auto&chunks=path==2?parallel:serial;
        const auto*src=path==0?page_in.data()+guard:pin_in.p+guard;
        auto*dst=path==0?page_out.data()+guard:pin_out.p+guard;
        last_count[0]=last_count[1]=0;
        for(const auto&q:chunks){
            auto stream=streams[q.slot].s;auto*in=din[q.slot]->p+guard;auto*out=dout[q.slot]->p+guard;
            CUDA(cudaMemcpyAsync(in,src+q.offset,q.count*sizeof(*src),cudaMemcpyHostToDevice,stream));
            affine<<<static_cast<unsigned>((q.count+255)/256),256,0,stream>>>(in,out,q.count);
            CUDA(cudaMemcpyAsync(dst+q.offset,out,q.count*sizeof(*dst),cudaMemcpyDeviceToHost,stream));
            last_offset[q.slot]=q.offset;last_count[q.slot]=q.count;
        }
        CUDA(cudaGetLastError());
        CUDA(cudaStreamSynchronize(streams[0].s));
        if(path==2)CUDA(cudaStreamSynchronize(streams[1].s));
    }
    void verify(int path){
        const auto*out=path==0?page_out.data():pin_out.p;
        if(!std::equal(expected.begin(),expected.end(),out+guard))throw std::runtime_error("independent oracle mismatch");
        for(const auto*p:{page_in.data(),page_out.data(),pin_in.p,pin_out.p}){
            for(std::size_t i=0;i<guard;++i)if(p[i]!=sentinel||p[i+n+guard]!=sentinel)throw std::runtime_error("host guard changed");
        }
        if(!std::equal(original.begin(),original.end(),page_in.begin()+guard)||!std::equal(original.begin(),original.end(),pin_in.p+guard))
            throw std::runtime_error("host input changed");
        for(int s=0;s<2;++s)for(int which=0;which<2;++which){
            std::vector<std::uint32_t>h(chunk+2*guard);
            CUDA(cudaMemcpy(h.data(),which?dout[s]->p:din[s]->p,h.size()*sizeof(h[0]),cudaMemcpyDeviceToHost));
            for(std::size_t i=0;i<guard;++i)if(h[i]!=sentinel||h[i+chunk+guard]!=sentinel)throw std::runtime_error("device guard changed");
            if(!which&&last_count[s]&&!std::equal(h.begin()+guard,h.begin()+guard+last_count[s],original.begin()+last_offset[s]))
                throw std::runtime_error("device input changed");
        }
    }
};
static void verify(bool quick){
    std::size_t cases=0;
    const auto sizes=quick?std::vector<std::size_t>{1,257,4097}:std::vector<std::size_t>{1,31,257,65537,1048579};
    const auto chunks=quick?std::vector<std::size_t>{256}:std::vector<std::size_t>{256,4096,65536};
    for(auto n:sizes)for(auto c:chunks){Work w(n,c);
        for(int p=0;p<3;++p){w.poison();w.run(p);w.verify(p);w.poison();w.run(p);w.verify(p);++cases;}
    }
    std::cout<<"PASS cases="<<cases<<" exact_oracle=pass poisoned_output=pass host_device_guards=pass input_immutable=pass repeated_slot_reuse=pass\n";
}
static void benchmark(){
    constexpr int warmup=3,repetitions=3,samples=9;
    std::cout<<"n,chunk,path,sample,repetitions,warmup,seed,host_ms,logical_GBps\n"<<std::setprecision(12);
    for(std::size_t n:{65537u,1048579u,4194307u})for(std::size_t c:{4096u,65536u}){
        Work w(n,c);
        for(int p=0;p<3;++p){w.poison();for(int k=0;k<warmup;++k)w.run(p);w.verify(p);}
        for(int sample=0;sample<samples;++sample)for(int order=0;order<3;++order){int p=(sample+order)%3;w.poison();
            auto t=std::chrono::steady_clock::now();
            for(int k=0;k<repetitions;++k)w.run(p);
            double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t).count()/repetitions;
            w.verify(p);
            std::cout<<n<<','<<c<<','<<names[p]<<','<<sample<<','<<repetitions<<','<<warmup<<",2027,"<<elapsed<<','<<8.0*n/(elapsed*1e6)<<'\n';
        }
    }
}
int main(int argc,char**argv){try{
    if(argc!=2)throw std::invalid_argument("expected one mode");
    std::string mode=argv[1];
    if(mode=="--help")std::cout<<"transfer_lab --verify | --quick | --benchmark | --diagnose | --profile\n";
    else if(mode=="--verify"||mode=="--quick")verify(mode=="--quick");
    else if(mode=="--benchmark")benchmark();
    else if(mode=="--profile"){Work w(1048579,65536);for(int p=1;p<3;++p){w.poison();w.run(p);w.verify(p);}}
    else if(mode=="--diagnose"){int d=0,rt=0,dr=0;cudaDeviceProp props{};
        CUDA(cudaGetDevice(&d));CUDA(cudaGetDeviceProperties(&props,d));CUDA(cudaRuntimeGetVersion(&rt));CUDA(cudaDriverGetVersion(&dr));
        std::cout<<"gpu="<<props.name<<" cc="<<props.major<<'.'<<props.minor<<" runtime="<<rt<<" driver_api="<<dr
                 <<" async_engine_count="<<props.asyncEngineCount<<" concurrent_kernels="<<props.concurrentKernels<<" block=256 seed=2027\n";
    }else throw std::invalid_argument("unknown mode");return 0;
}catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<'\n';return 1;}}
