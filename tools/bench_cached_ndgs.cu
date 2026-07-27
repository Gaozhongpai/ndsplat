// Cached-inverse N-DGS slicing benchmark (NeurIPS 2026 rebuttal, Reviewer HuLf W6).
//
// HuLf asks whether the reported 6.9-7.7x slicing speedup survives against an
// N-DGS implementation that caches the query-block inverse and the regression
// operator at inference time, when the trained parameters are fixed.
//
// Four variants, identical harness, timed at N primitives:
//
//   1. ndgs_onthefly  -- what the paper measures. Loads the DxD joint covariance,
//                        inverts the CxC query block (cofactor for C=3, Gauss-Jordan
//                        otherwise), forms v_regr = v12 @ inv, applies the Schur
//                        correction, and evaluates opacity.
//   2. ndgs_cached    -- reads precomputed M [3xC], Sigma_cond [6] and inv22 [CxC].
//                        No DxD load, no inversion, no regression product, no
//                        correction. This is the strongest inference-time N-DGS.
//   3. ndgs_rss       -- Sigma_qq = R S S^T R^T, so the inverse is closed-form from
//                        the factors; still forms v_regr and the correction.
//   4. dgs_direct     -- the direct kernel: z = L^T delta, opacity from |z|^2, and
//                        position from v_12 diag(Lambda) V_qq delta.
//
// Build (inside the container):
//   nvcc -O3 -arch=sm_80 -o /tmp/bench_cached tools/bench_cached_ndgs.cu
//   /tmp/bench_cached 1000000 100 20
//
// Reported time is the median over `trials` after `warmup`, measured with CUDA
// events. Only the slicing kernel is timed -- no rasterization, no autograd.

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <vector>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ \
    printf("CUDA error %s at %d\n", cudaGetErrorString(e), __LINE__); exit(1);} } while(0)

__device__ __forceinline__ void invert3x3(const float* a, float* o) {
    float c00 =  a[4]*a[8]-a[5]*a[7], c01 = -(a[3]*a[8]-a[5]*a[6]), c02 =  a[3]*a[7]-a[4]*a[6];
    float c10 = -(a[1]*a[8]-a[2]*a[7]), c11 =  a[0]*a[8]-a[2]*a[6], c12 = -(a[0]*a[7]-a[1]*a[6]);
    float c20 =  a[1]*a[5]-a[2]*a[4], c21 = -(a[0]*a[5]-a[2]*a[3]), c22 =  a[0]*a[4]-a[1]*a[3];
    float det = a[0]*c00 + a[1]*c01 + a[2]*c02;
    if (det == 0.f) det = 1e-20f;
    float id = 1.f/det;
    o[0]=c00*id; o[1]=c10*id; o[2]=c20*id;
    o[3]=c01*id; o[4]=c11*id; o[5]=c21*id;
    o[6]=c02*id; o[7]=c12*id; o[8]=c22*id;
}

// Register-local Gauss-Jordan for small CxC (used when C != 3).
template <int C>
__device__ __forceinline__ void invertGJ(const float* in, float* out) {
    float a[C*C], b[C*C];
    #pragma unroll
    for (int i=0;i<C*C;++i) { a[i]=in[i]; b[i]=0.f; }
    #pragma unroll
    for (int i=0;i<C;++i) b[i*C+i]=1.f;
    #pragma unroll
    for (int col=0; col<C; ++col) {
        float piv = a[col*C+col];
        if (piv == 0.f) piv = 1e-20f;
        float ip = 1.f/piv;
        #pragma unroll
        for (int j=0;j<C;++j) { a[col*C+j]*=ip; b[col*C+j]*=ip; }
        #pragma unroll
        for (int r=0;r<C;++r) if (r!=col) {
            float f = a[r*C+col];
            #pragma unroll
            for (int j=0;j<C;++j) { a[r*C+j]-=f*a[col*C+j]; b[r*C+j]-=f*b[col*C+j]; }
        }
    }
    #pragma unroll
    for (int i=0;i<C*C;++i) out[i]=b[i];
}

// ---- 1. N-DGS on the fly: the operation the paper measures ------------------
template <int C>
__global__ void ndgs_onthefly(int N, const float* __restrict__ m1,
                              const float* __restrict__ m2, const float* __restrict__ q,
                              const float* __restrict__ cov, float lam,
                              float* __restrict__ o_mu, float* __restrict__ o_cov,
                              float* __restrict__ o_sc) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
    constexpr int D = 3+C;
    const float* v = cov + (size_t)i*D*D;
    float x[C];
    #pragma unroll
    for (int c=0;c<C;++c) x[c] = q[(size_t)i*C+c] - m2[(size_t)i*C+c];
    float v11[9], v12[3*C], v21[C*3], v22[C*C];
    #pragma unroll
    for (int r=0;r<3;++r) { for (int c=0;c<3;++c) v11[r*3+c]=v[r*D+c];
                            for (int c=0;c<C;++c) v12[r*C+c]=v[r*D+3+c]; }
    #pragma unroll
    for (int r=0;r<C;++r) { for (int c=0;c<3;++c) v21[r*3+c]=v[(3+r)*D+c];
                            for (int c=0;c<C;++c) v22[r*C+c]=v[(3+r)*D+3+c]; }
    float inv[C*C];
    if (C==3) invert3x3(v22, inv); else invertGJ<C>(v22, inv);
    float regr[3*C];
    #pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<C;++c) {
        float a=0.f;
        #pragma unroll
        for (int k=0;k<C;++k) a += v12[r*C+k]*inv[k*C+c];
        regr[r*C+c]=a;
    }
    #pragma unroll
    for (int r=0;r<3;++r) {
        float a = m1[(size_t)i*3+r];
        #pragma unroll
        for (int c=0;c<C;++c) a += regr[r*C+c]*x[c];
        o_mu[(size_t)i*3+r]=a;
    }
    #pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
        float a=v11[r*3+c];
        #pragma unroll
        for (int k=0;k<C;++k) a -= regr[r*C+k]*v21[k*3+c];
        o_cov[(size_t)i*9+r*3+c]=a;
    }
    float quad=0.f;
    #pragma unroll
    for (int r=0;r<C;++r) { float a=0.f;
        #pragma unroll
        for (int c=0;c<C;++c) a += inv[r*C+c]*x[c];
        quad += x[r]*a; }
    o_sc[i]=__expf(-lam*quad);
}

// ---- 2. N-DGS cached: M, Sigma_cond and inv22 precomputed -------------------
template <int C>
__global__ void ndgs_cached(int N, const float* __restrict__ m1,
                            const float* __restrict__ m2, const float* __restrict__ q,
                            const float* __restrict__ M,      // [N,3C]
                            const float* __restrict__ Scond,  // [N,6] lower-tri
                            const float* __restrict__ Inv22,  // [N,C*C]
                            float lam,
                            float* __restrict__ o_mu, float* __restrict__ o_cov,
                            float* __restrict__ o_sc) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
    float x[C];
    #pragma unroll
    for (int c=0;c<C;++c) x[c] = q[(size_t)i*C+c] - m2[(size_t)i*C+c];
    // position: one [3xC] x [C] product, no inversion, no regression build
    #pragma unroll
    for (int r=0;r<3;++r) {
        float a = m1[(size_t)i*3+r];
        #pragma unroll
        for (int c=0;c<C;++c) a += M[(size_t)i*3*C + r*C + c]*x[c];
        o_mu[(size_t)i*3+r]=a;
    }
    // covariance: straight copy of the cached conditional, no Schur correction
    const float* s = Scond + (size_t)i*6;
    o_cov[(size_t)i*9+0]=s[0]; o_cov[(size_t)i*9+1]=s[1]; o_cov[(size_t)i*9+2]=s[2];
    o_cov[(size_t)i*9+3]=s[1]; o_cov[(size_t)i*9+4]=s[3]; o_cov[(size_t)i*9+5]=s[4];
    o_cov[(size_t)i*9+6]=s[2]; o_cov[(size_t)i*9+7]=s[4]; o_cov[(size_t)i*9+8]=s[5];
    // opacity: cached inverse, so only the quadratic form remains
    float quad=0.f;
    #pragma unroll
    for (int r=0;r<C;++r) { float a=0.f;
        #pragma unroll
        for (int c=0;c<C;++c) a += Inv22[(size_t)i*C*C + r*C+c]*x[c];
        quad += x[r]*a; }
    o_sc[i]=__expf(-lam*quad);
}

// ---- 3. N-DGS with Sigma_qq = R S S^T R^T (closed-form inverse) -------------
template <int C>
__global__ void ndgs_rss(int N, const float* __restrict__ m1,
                         const float* __restrict__ m2, const float* __restrict__ q,
                         const float* __restrict__ v12in,   // [N,3C]
                         const float* __restrict__ v11in,   // [N,6]
                         const float* __restrict__ Rq,      // [N,C*C] rotation
                         const float* __restrict__ Sq,      // [N,C]   scales
                         float lam,
                         float* __restrict__ o_mu, float* __restrict__ o_cov,
                         float* __restrict__ o_sc) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
    float x[C];
    #pragma unroll
    for (int c=0;c<C;++c) x[c] = q[(size_t)i*C+c] - m2[(size_t)i*C+c];
    // inv(R S S^T R^T) = R diag(1/s^2) R^T -- no general inversion needed
    float inv[C*C];
    #pragma unroll
    for (int r=0;r<C;++r) for (int c=0;c<C;++c) {
        float a=0.f;
        #pragma unroll
        for (int k=0;k<C;++k) {
            float s = Sq[(size_t)i*C+k];
            a += Rq[(size_t)i*C*C + r*C+k] * (1.f/(s*s)) * Rq[(size_t)i*C*C + c*C+k];
        }
        inv[r*C+c]=a;
    }
    float v12[3*C];
    #pragma unroll
    for (int j=0;j<3*C;++j) v12[j]=v12in[(size_t)i*3*C+j];
    float regr[3*C];
    #pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<C;++c) {
        float a=0.f;
        #pragma unroll
        for (int k=0;k<C;++k) a += v12[r*C+k]*inv[k*C+c];
        regr[r*C+c]=a;
    }
    #pragma unroll
    for (int r=0;r<3;++r) {
        float a=m1[(size_t)i*3+r];
        #pragma unroll
        for (int c=0;c<C;++c) a += regr[r*C+c]*x[c];
        o_mu[(size_t)i*3+r]=a;
    }
    const float* s6 = v11in + (size_t)i*6;
    float v11[9] = {s6[0],s6[1],s6[2], s6[1],s6[3],s6[4], s6[2],s6[4],s6[5]};
    #pragma unroll
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) {
        float a=v11[r*3+c];
        #pragma unroll
        for (int k=0;k<C;++k) a -= regr[r*C+k]*v12[c*C+k];
        o_cov[(size_t)i*9+r*3+c]=a;
    }
    float quad=0.f;
    #pragma unroll
    for (int r=0;r<C;++r) { float a=0.f;
        #pragma unroll
        for (int c=0;c<C;++c) a += inv[r*C+c]*x[c];
        quad += x[r]*a; }
    o_sc[i]=__expf(-lam*quad);
}

// ---- 4. dGS direct ---------------------------------------------------------
template <int C>
__global__ void dgs_direct(int N, const float* __restrict__ xyz,
                           const float* __restrict__ m2, const float* __restrict__ q,
                           const float* __restrict__ v12,   // [N,3C]
                           const float* __restrict__ Lraw,  // [N,C(C+1)/2]
                           const float* __restrict__ lamv,  // [N]
                           float lam_opc,
                           float* __restrict__ o_mu, float* __restrict__ o_sc) {
    int i = blockIdx.x*blockDim.x + threadIdx.x; if (i>=N) return;
    constexpr int LS = C*(C+1)/2;
    float d[C];
    #pragma unroll
    for (int c=0;c<C;++c) d[c] = q[(size_t)i*C+c] - m2[(size_t)i*C+c];
    float L[C][C];
    #pragma unroll
    for (int r=0;r<C;++r)
        #pragma unroll
        for (int c=0;c<C;++c) L[r][c]=0.f;
    int k=0;
    #pragma unroll
    for (int r=0;r<C;++r)
        #pragma unroll
        for (int c=0;c<=r;++c) { float t=Lraw[(size_t)i*LS+k];
            L[r][c] = (r==c) ? __expf(t) : t; ++k; }
    float z[C];
    #pragma unroll
    for (int c=0;c<C;++c) { float a=0.f;
        #pragma unroll
        for (int r=0;r<C;++r) a += L[r][c]*d[r];
        z[c]=a; }
    float q2=0.f;
    #pragma unroll
    for (int c=0;c<C;++c) q2 += z[c]*z[c];
    o_sc[i]=__expf(-lam_opc*q2);
    float Vqq[C][C];
    #pragma unroll
    for (int r=0;r<C;++r)
        #pragma unroll
        for (int c=0;c<C;++c) { float a=0.f;
            #pragma unroll
            for (int t=0;t<C;++t) a += L[r][t]*L[c][t];
            Vqq[r][c]=a; }
    float Vd[C];
    #pragma unroll
    for (int r=0;r<C;++r) { float a=0.f;
        #pragma unroll
        for (int c=0;c<C;++c) a += Vqq[r][c]*d[c];
        Vd[r]=a; }
    float lv = lamv[i];
    #pragma unroll
    for (int r=0;r<3;++r) {
        float a = xyz[(size_t)i*3+r];
        #pragma unroll
        for (int c=0;c<C;++c) a += v12[(size_t)i*3*C + r*C + c]*lv*Vd[c];
        o_mu[(size_t)i*3+r]=a;
    }
}

static float* dev_rand(size_t n, unsigned seed) {
    std::vector<float> h(n);
    srand(seed);
    for (size_t i=0;i<n;++i) h[i] = (float)rand()/RAND_MAX - 0.5f;
    float* d; CK(cudaMalloc(&d, n*sizeof(float)));
    CK(cudaMemcpy(d, h.data(), n*sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

// SPD joint covariance so the on-the-fly inverse is well conditioned.
template <int C>
static float* dev_spd(int N, unsigned seed) {
    constexpr int D = 3+C;
    std::vector<float> h((size_t)N*D*D);
    srand(seed);
    for (int i=0;i<N;++i) {
        float A[D*D];
        for (int j=0;j<D*D;++j) A[j] = ((float)rand()/RAND_MAX - 0.5f)*0.3f;
        for (int r=0;r<D;++r) for (int c=0;c<D;++c) {
            float a=0.f; for (int k=0;k<D;++k) a += A[r*D+k]*A[c*D+k];
            h[(size_t)i*D*D + r*D+c] = a + (r==c ? 1.0f : 0.f);
        }
    }
    float* d; CK(cudaMalloc(&d, h.size()*sizeof(float)));
    CK(cudaMemcpy(d, h.data(), h.size()*sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

template <typename F>
static float time_ms(F launch, int trials, int warmup) {
    cudaEvent_t a,b; CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
    for (int i=0;i<warmup;++i) launch();
    CK(cudaDeviceSynchronize());
    std::vector<float> ts;
    for (int i=0;i<trials;++i) {
        CK(cudaEventRecord(a));
        launch();
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms; CK(cudaEventElapsedTime(&ms,a,b)); ts.push_back(ms);
    }
    std::sort(ts.begin(), ts.end());
    return ts[ts.size()/2];
}

template <int C>
static void run(int N, int trials, int warmup) {
    constexpr int D = 3+C;
    const int TH = 256, BL = (N+TH-1)/TH;
    float *m1=dev_rand((size_t)N*3,1), *m2=dev_rand((size_t)N*C,2), *q=dev_rand((size_t)N*C,3);
    float *cov=dev_spd<C>(N,4);
    float *M=dev_rand((size_t)N*3*C,5), *Sc=dev_rand((size_t)N*6,6), *Inv=dev_spd<C>(N,7);
    float *Rq=dev_rand((size_t)N*C*C,8), *Sq=dev_rand((size_t)N*C,9);
    float *v12=dev_rand((size_t)N*3*C,10), *Lraw=dev_rand((size_t)N*C*(C+1)/2,11), *lamv=dev_rand(N,12);
    float *o_mu,*o_cov,*o_sc;
    CK(cudaMalloc(&o_mu,(size_t)N*3*sizeof(float)));
    CK(cudaMalloc(&o_cov,(size_t)N*9*sizeof(float)));
    CK(cudaMalloc(&o_sc,(size_t)N*sizeof(float)));

    float t1 = time_ms([&]{ ndgs_onthefly<C><<<BL,TH>>>(N,m1,m2,q,cov,0.35f,o_mu,o_cov,o_sc); }, trials, warmup);
    float t2 = time_ms([&]{ ndgs_cached<C><<<BL,TH>>>(N,m1,m2,q,M,Sc,Inv,0.35f,o_mu,o_cov,o_sc); }, trials, warmup);
    float t3 = time_ms([&]{ ndgs_rss<C><<<BL,TH>>>(N,m1,m2,q,v12,Sc,Rq,Sq,0.35f,o_mu,o_cov,o_sc); }, trials, warmup);
    float t4 = time_ms([&]{ dgs_direct<C><<<BL,TH>>>(N,m1,m2,q,v12,Lraw,lamv,0.35f,o_mu,o_sc); }, trials, warmup);

    printf("\n=== C=%d (D=%d), N=%d ===\n", C, D, N);
    printf("  %-34s %8.4f ms   %6.2fx vs dGS\n", "N-DGS on-the-fly", t1, t1/t4);
    printf("  %-34s %8.4f ms   %6.2fx vs dGS\n", "N-DGS cached (inv + M + Scond)", t2, t2/t4);
    printf("  %-34s %8.4f ms   %6.2fx vs dGS\n", "N-DGS RSS^T R^T closed-form inv", t3, t3/t4);
    printf("  %-34s %8.4f ms   %6.2fx (ref)\n", "dGS direct", t4, 1.0f);
    printf("  -- cached vs on-the-fly speedup: %.2fx\n", t1/t2);
    printf("  -- cached state: %d floats/primitive = %.1f MB per 1M primitives\n",
           C*C + 3*C + 6, (C*C + 3*C + 6)*4.0);
    printf("  -- on-the-fly reads the DxD joint covariance: %d floats/primitive = %.1f MB per 1M\n",
           D*D, D*D*4.0);

    cudaFree(m1);cudaFree(m2);cudaFree(q);cudaFree(cov);cudaFree(M);cudaFree(Sc);
    cudaFree(Inv);cudaFree(Rq);cudaFree(Sq);cudaFree(v12);cudaFree(Lraw);cudaFree(lamv);
    cudaFree(o_mu);cudaFree(o_cov);cudaFree(o_sc);
}

int main(int argc, char** argv) {
    int N = argc>1 ? atoi(argv[1]) : 1000000;
    int trials = argc>2 ? atoi(argv[2]) : 100;
    int warmup = argc>3 ? atoi(argv[3]) : 20;
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p,0));
    printf("Device: %s\nN=%d  trials=%d  warmup=%d\n", p.name, N, trials, warmup);
    run<3>(N, trials, warmup);
    run<4>(N, trials, warmup);
    return 0;
}
