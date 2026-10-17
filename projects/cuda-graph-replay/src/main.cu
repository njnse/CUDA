#include <cuda_runtime.h>
#include "reference.hpp"
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
void ck(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
using Clock=std::chrono::steady_clock;
double us(Clock::time_point a,Clock::time_point b){return std::chrono::duration<double,std::micro>(b-a).count();}
constexpr int GUARD=16;
constexpr float SENTINEL=-1234567.0f;
__global__ void affine(const float*x,float*t,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)t[i]=1.25f*x[i]-0.5f;}
__global__ void relu(const float*t,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=fmaxf(t[i],0.0f);}
__global__ void fused(const float*x,float*y,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)y[i]=fmaxf(1.25f*x[i]-0.5f,0.0f);}
struct Device {
    float*p=nullptr;
    explicit Device(size_t n){ck(cudaMalloc(reinterpret_cast<void**>(&p),n*sizeof(float)));}
    ~Device(){if(p)cudaFree(p);}
    Device(const Device&)=delete;
};
void experiment(int n,bool bench) {
    if(n<1)throw std::invalid_argument("n must be positive");
    size_t bytes=size_t(n)*sizeof(float), guarded=size_t(n)+2*GUARD;
    std::vector<float>x(n),y(guarded),init(guarded,SENTINEL),back(n);
    for(int i=0;i<n;++i)x[i]=float((i%2049)-1024)/256.0f;
    auto expected=reference(x);
    Device input(n),temp(guarded),output(guarded);

    cudaStream_t stream;ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    ck(cudaMemcpyAsync(input.p,x.data(),bytes,cudaMemcpyHostToDevice,stream));
    ck(cudaStreamSynchronize(stream));
    auto eager=[&] {
        affine<<<(n+255)/256,256,0,stream>>>(input.p,temp.p+GUARD,n);ck(cudaGetLastError());
        relu<<<(n+255)/256,256,0,stream>>>(temp.p+GUARD,output.p+GUARD,n);ck(cudaGetLastError());
    };
    cudaGraph_t graph;cudaGraphExec_t executable;
    auto setup_start=Clock::now();
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    eager();ck(cudaStreamEndCapture(stream,&graph));
    auto capture_end=Clock::now();
    size_t nodes=0;ck(cudaGraphGetNodes(graph,nullptr,&nodes));
    if(nodes!=2)throw std::runtime_error("expected exactly two graph nodes");
    ck(cudaGraphInstantiate(&executable,graph,nullptr,nullptr,0));
    auto instantiate_end=Clock::now();
    ck(cudaGraphUpload(executable,stream));ck(cudaStreamSynchronize(stream));
    auto upload_end=Clock::now();
    std::cerr<<"SETUP n="<<n<<" capture_us="<<us(setup_start,capture_end)
             <<" instantiate_us="<<us(capture_end,instantiate_end)
             <<" upload_us="<<us(instantiate_end,upload_end)
             <<" total_us="<<us(setup_start,upload_end)<<" nodes="<<nodes<<"\n";
    auto launch=[&](int mode) {
        if(mode==0)eager();
        else if(mode==1)ck(cudaGraphLaunch(executable,stream));
        else {fused<<<(n+255)/256,256,0,stream>>>(input.p,output.p+GUARD,n);ck(cudaGetLastError());}
    };
    const char*names[]={"eager","graph","fused"};
    for(int mode=0;mode<3;++mode) {
        std::fill(init.begin(),init.end(),SENTINEL);
        std::fill(init.begin()+GUARD,init.begin()+GUARD+n,std::numeric_limits<float>::quiet_NaN());
        ck(cudaMemcpyAsync(temp.p,init.data(),guarded*sizeof(float),cudaMemcpyHostToDevice,stream));
        ck(cudaMemcpyAsync(output.p,init.data(),guarded*sizeof(float),cudaMemcpyHostToDevice,stream));
        launch(mode);ck(cudaStreamSynchronize(stream));
        ck(cudaMemcpy(y.data(),output.p,guarded*sizeof(float),cudaMemcpyDeviceToHost));
        for(int i=0;i<n;++i)if(y[i+GUARD]!=expected[i])throw std::runtime_error(std::string(names[mode])+" mismatch at "+std::to_string(i)+" actual="+std::to_string(y[i+GUARD])+" expected="+std::to_string(expected[i]));
        for(int g=0;g<GUARD;++g)if(y[g]!=SENTINEL||y[n+GUARD+g]!=SENTINEL)throw std::runtime_error("output guard");
        ck(cudaMemcpy(y.data(),temp.p,guarded*sizeof(float),cudaMemcpyDeviceToHost));
        for(int g=0;g<GUARD;++g)if(y[g]!=SENTINEL||y[n+GUARD+g]!=SENTINEL)throw std::runtime_error("temporary guard");
    }
    ck(cudaMemcpy(back.data(),input.p,bytes,cudaMemcpyDeviceToHost));
    if(back!=x)throw std::runtime_error("input modified");
    std::cerr<<"PASS n="<<n<<" modes=3 exact_reference=pass guards=pass input_immutable=pass\n";
    if(bench) {
        for(int mode=0;mode<3;++mode){for(int w=0;w<20;++w)launch(mode);ck(cudaStreamSynchronize(stream));}
        cudaEvent_t start,stop;ck(cudaEventCreate(&start));ck(cudaEventCreate(&stop));
        int repeats=n<(1<<20)?200:30;
        for(int sample=0;sample<15;++sample)for(int position=0;position<3;++position) {
            int mode=(sample+position)%3;
            auto wall_start=Clock::now();ck(cudaEventRecord(start,stream));
            auto submit_start=Clock::now();
            for(int r=0;r<repeats;++r)launch(mode);
            auto submit_end=Clock::now();
            ck(cudaEventRecord(stop,stream));ck(cudaEventSynchronize(stop));auto wall_end=Clock::now();
            float ms=0;ck(cudaEventElapsedTime(&ms,start,stop));
            if(ms<=0)throw std::runtime_error("nonpositive device interval");
            std::cout<<n<<','<<names[mode]<<','<<sample<<','<<position<<','<<repeats<<','
                     <<double(ms)*1000/repeats<<','<<us(wall_start,wall_end)/repeats<<','
                     <<us(submit_start,submit_end)/repeats<<'\n';
        }
        ck(cudaEventDestroy(start));ck(cudaEventDestroy(stop));
    }
    ck(cudaGraphExecDestroy(executable));ck(cudaGraphDestroy(graph));ck(cudaStreamDestroy(stream));
}
int main(int argc,char**argv) {
    try {
        std::string mode=argc==2?argv[1]:"--help";
        if(mode=="--help"){std::cout<<"graph_lab --verify | --benchmark\n";return 0;}
        if(mode!="--verify"&&mode!="--benchmark")throw std::invalid_argument("unknown mode");
        int count=0;cudaError_t status=cudaGetDeviceCount(&count);
        if(status!=cudaSuccess||count==0){std::cerr<<"GPU_UNAVAILABLE: "<<cudaGetErrorString(status)<<"\n";return 77;}
        cudaDeviceProp prop;ck(cudaGetDeviceProperties(&prop,0));
        int rt=0,driver=0;ck(cudaRuntimeGetVersion(&rt));ck(cudaDriverGetVersion(&driver));
        std::cerr<<std::setprecision(10)<<"GPU="<<prop.name<<" cc="<<prop.major<<'.'<<prop.minor
                 <<" runtime="<<rt<<" driver_api="<<driver<<"\n";
        std::cout<<std::setprecision(10);
        if(mode=="--benchmark")std::cout<<"n,mode,sample,position,iterations,gpu_us,wall_us,submit_us\n";
        for(int n:{1,31,257,4096,1048576})experiment(n,mode=="--benchmark");
        return 0;
    }catch(const std::exception&e){std::cerr<<"ERROR: "<<e.what()<<"\n";return 1;}
}
