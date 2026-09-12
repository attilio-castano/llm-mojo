// Process-local diagnostic for the pinned MAX 26.5.0 macOS arm64 C ABI.
// No GPU operations or arguments are changed. DYLD interposition applies only
// to the benchmark child process that explicitly loads this library.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdatomic.h>
#include <pthread.h>
#define LIMIT 131072
#define PARAMS void *ctx, void *fn, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz, uint32_t shared, void *attrs, uint32_t nattrs, void **args, uint32_t nargs, uint64_t *sizes
#define ARGS ctx,fn,gx,gy,gz,bx,by,bz,shared,attrs,nattrs,args,nargs,sizes
extern const char *AsyncRT_DeviceContext_enqueueFunctionDirect(PARAMS);
typedef struct { uint64_t begin,end,thread; uint32_t gx,gy,gz,bx,by,bz,nargs,error; } Record;
static Record records[LIMIT];
static _Atomic uint32_t count;
static const char *path;
static int enabled;
static _Thread_local uint64_t thread_id;
__attribute__((constructor)) static void init(void) {
    path=getenv("LLM_MOJO_ENQUEUE_RECORD");enabled=path && *path;
    if(enabled) { volatile char *p=(volatile char *)records; for(size_t i=0;i<sizeof(records);i+=4096) p[i]=0; }
}
static const char *probe(PARAMS) {
    if(!enabled) return AsyncRT_DeviceContext_enqueueFunctionDirect(ARGS);
    if(!thread_id) pthread_threadid_np(NULL,&thread_id);
    uint64_t begin=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    const char *result=AsyncRT_DeviceContext_enqueueFunctionDirect(ARGS);
    uint64_t end=clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
    uint32_t index=atomic_fetch_add_explicit(&count,1,memory_order_relaxed);
    if(index<LIMIT) records[index]=(Record){begin,end,thread_id,gx,gy,gz,bx,by,bz,nargs,result!=NULL};
    return result;
}
__attribute__((used)) static const struct { const void *replacement,*original; } interpose
__attribute__((section("__DATA,__interpose")))={(const void *)probe,(const void *)AsyncRT_DeviceContext_enqueueFunctionDirect};
__attribute__((destructor)) static void finish(void) {
    if(!enabled) return;
    FILE *f=fopen(path,"w");if(!f) { perror("enqueue probe output"); return; }
    uint32_t n=atomic_load_explicit(&count,memory_order_relaxed);
    fprintf(f,"ENQUEUE_PROBE_V1 %u %u\n",n,n>LIMIT?n-LIMIT:0);
    for(uint32_t i=0;i<n && i<LIMIT;i++) {
        Record *r=&records[i];
        fprintf(f,"%llu %llu %llu %u %u %u %u %u %u %u %u\n",r->begin,r->end,r->thread,r->gx,r->gy,r->gz,r->bx,r->by,r->bz,r->nargs,r->error);
    }
    fclose(f);
}
