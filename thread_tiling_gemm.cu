#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>



// ============================================================
// CUDA Error Check
// ============================================================
#define CUDA_CHECK(call)                                                     \
do {                                                                         \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
        fprintf(stderr,                                                       \
                "CUDA Error [%s:%d]: %s\n",                                  \
                __FILE__,                                                    \
                __LINE__,                                                    \
                cudaGetErrorString(err));                                    \
        exit(EXIT_FAILURE);                                                  \
    }                                                                        \
} while(0)





// ============================================================
// Thread Tiling SGEMM
//
// 优化层次:
//
// Global Memory
//        |
//        v
// Shared Memory Tile
//        |
//        v
// Warp Mapping
//        |
//        v
// Register Tile
//
//
// 一个 thread:
//      TM × TN
//
// 一个 block:
//      BM × BN
//
// ============================================================

template<int BM, int BK>
__device__ inline void load_tile_A(
    const float* __restrict__ A,
    float As[BM][BK],
    int block_row,
    int bk,
    int tid,
    int M,
    int K)
{
    int tile_size = BM * BK;

    for (int i = tid; i < tile_size; i += blockDim.x)
    {
        int row = i / BK;
        int col = i % BK;

        int global_row = block_row * BM + row;
        int global_col = bk + col;


        if (global_row < M && global_col < K)
            As[row][col] = A[global_row * K + global_col];
        else
            As[row][col] = 0.0f;
    }
}


template<int BK, int BN>
__device__ inline void load_tile_B(
    const float* __restrict__ B,
    float Bs[BK][BN],
    int block_col,
    int bk,
    int tid,
    int K,
    int N)
{
    int tile_size = BK * BN;


    for (int i = tid; i < tile_size; i += blockDim.x)
    {
        int row = i / BN;
        int col = i % BN;


        int global_row = bk + row;
        int global_col = block_col * BN + col;


        if (global_row < K && global_col < N)
            Bs[row][col] = B[global_row * N + global_col];
        else
            Bs[row][col] = 0.0f;
    }
}


template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN
>
__global__ void thread_tiling_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{


    // ========================================================
    // Shared Memory
    // ========================================================
    __shared__ float As[BM][BK];

    __shared__ float Bs[BK][BN];


    const int tid =
        threadIdx.x;



    // ========================================================
    // Warp / Lane Mapping
    //
    // Block: 4 × 2 warps
    // Warp: 4 × 8 lanes
    //
    // 每个 thread:
    // TM × TN register tile
    //
    // ========================================================

    int warp_id = tid >> 5;      // tid / 32
    int lane_id = tid & 31;      // tid % 32


    // warp 在 block tile 中的位置
    // warp_row: M方向
    // warp_col: N方向
    int warp_row = warp_id >> 1; // warp_id / 2
    int warp_col = warp_id & 1;  // warp_id % 2


    // lane 在 warp 内的位置
    // 4 × 8 排列
    int lane_row = lane_id >> 3; // lane_id / 8
    int lane_col = lane_id & 7;  // lane_id % 8



    // thread负责的register tile起点
    const int thread_row =
        (warp_row * 4 + lane_row) * TM;


    const int thread_col =
        (warp_col * 8 + lane_col) * TN;



    // ========================================================
    // Register Tile
    // ========================================================
    float reg_a[TM];

    float reg_b[TN];


    float reg_c[TM][TN] =
        {0.0f};



    const int block_row =
        blockIdx.y;


    const int block_col =
        blockIdx.x;


    for(int bk = 0;
        bk < K;
        bk += BK)
    {
    
    // 协作加载 A、B 到 Shared Memory（省略边界检查）
    load_tile_A<BM, BK>(A, As, block_row, bk, tid, M, K);
    load_tile_B<BK, BN>(B, Bs, block_col, bk, tid, K, N);


    __syncthreads();




    // ====================================================
    // Compute
    //
    // Shared Memory
    //        |
    //        v
    // Register
    //
    // Outer Product
    //
    // ====================================================

    #pragma unroll
    for(int k = 0;
        k < BK;
        k++)
    {


        #pragma unroll
        for(int i=0;
            i<TM;
            i++)
        {
            reg_a[i] =
                As[
                    thread_row+i
                ][k];
        }



        #pragma unroll
        for(int j=0;
            j<TN;
            j++)
        {
            reg_b[j] =
                Bs[k]
                [
                    thread_col+j
                ];
        }




        #pragma unroll
        for(int i=0;
            i<TM;
            i++)
        {

            #pragma unroll
            for(int j=0;
                j<TN;
                j++)
            {

                reg_c[i][j] +=
                    reg_a[i] *
                    reg_b[j];

            }
        }

    }


    __syncthreads();

    }




    // ========================================================
    // Store C
    // ========================================================

    #pragma unroll
    for(int i=0;
        i<TM;
        i++)
    {

        #pragma unroll
        for(int j=0;
            j<TN;
            j++)
        {

            C[(block_row * BM + thread_row + i) * N + block_col * BN + thread_col + j] = reg_c[i][j];
           

        }
    }

}






// ============================================================
// Matrix Initialize
// ============================================================
void fill_random(float* data,int size)
{
    for(int i=0;i<size;i++)
    {
        data[i]=
            static_cast<float>(rand())
            /
            RAND_MAX;
    }
}





// ============================================================
// CPU Reference
// ============================================================
void sgemm_cpu(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K)
{

    for(int i=0;i<M;i++)
    {
        for(int j=0;j<N;j++)
        {

            float sum=0.0f;


            for(int k=0;k<K;k++)
            {
                sum +=
                    A[i*K+k] *
                    B[k*N+j];
            }


            C[i*N+j]=sum;
        }
    }
}




// ============================================================
// Verify
// ============================================================
bool verify_result(
    float* gpu,
    float* cpu,
    int size)
{

    for(int i=0;i<size;i++)
    {

        if(fabs(gpu[i]-cpu[i])>1e-3)
        {
            printf(
                "Mismatch %d GPU=%f CPU=%f\n",
                i,
                gpu[i],
                cpu[i]);

            return false;
        }
    }


    return true;
}






// ============================================================
// Main
// ============================================================
int main()
{

    constexpr int M=1024;
    constexpr int N=1024;
    constexpr int K=1024;



    constexpr int BM=128;
    constexpr int BN=64;
    constexpr int BK=32;


    constexpr int TM=8;
    constexpr int TN=4;



    size_t size_A =
        M*K*sizeof(float);


    size_t size_B =
        K*N*sizeof(float);


    size_t size_C =
        M*N*sizeof(float);



    float* h_A =
        (float*)malloc(size_A);


    float* h_B =
        (float*)malloc(size_B);


    float* h_C =
        (float*)malloc(size_C);


    float* h_ref =
        (float*)malloc(size_C);



    fill_random(h_A,M*K);

    fill_random(h_B,K*N);



    float *d_A,*d_B,*d_C;



    CUDA_CHECK(cudaMalloc(&d_A,size_A));

    CUDA_CHECK(cudaMalloc(&d_B,size_B));

    CUDA_CHECK(cudaMalloc(&d_C,size_C));



    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,size_A,
        cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        size_B,
        cudaMemcpyHostToDevice));




    dim3 block(256);


    dim3 grid(
        (N+BN-1)/BN,
        (M+BM-1)/BM);



    printf(
        "Thread Tiling GEMM\n");

    printf(
        "BM=%d BN=%d BK=%d TM=%d TN=%d\n",
        BM,BN,BK,TM,TN);



    // warmup

    thread_tiling_gemm<
        BM,
        BN,
        BK,
        TM,
        TN>
    <<<grid,block>>>(
        d_A,
        d_B,
        d_C,
        M,
        N,
        K);



    CUDA_CHECK(cudaDeviceSynchronize());



    cudaEvent_t start,stop;


    CUDA_CHECK(cudaEventCreate(&start));

    CUDA_CHECK(cudaEventCreate(&stop));



    constexpr int RUNS=10;


    float total_ms=0;



    for(int i=0;i<RUNS;i++)
    {

        cudaEventRecord(start);


        thread_tiling_gemm<
            BM,
            BN,
            BK,
            TM,
            TN>
        <<<grid,block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K);


        CUDA_CHECK(cudaGetLastError());


        cudaEventRecord(stop);


        cudaEventSynchronize(stop);


        float ms;


        cudaEventElapsedTime(
            &ms,start,stop);


        total_ms+=ms;

    }



    float avg_ms =
        total_ms/RUNS;



    double flops =
        2.0*M*N*K;



    double gflops =
        flops/
        (avg_ms*1e6);



    printf("============================\n");
    printf("Matrix : %d x %d x %d\n",
           M,
           N,
           K);

    printf(
        "Time : %.3f ms\n",
        avg_ms);


    printf(
        "GFLOPS : %.2f\n",
        gflops);

  printf("============================\n");


    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        size_C,
        cudaMemcpyDeviceToHost));



    sgemm_cpu(
        h_A,
        h_B,
        h_ref,
        M,N,K);



    if(verify_result(
        h_C,
        h_ref,
        M*N))
    {
        printf(
            "Verification PASSED\n");
    }
    else
    {
        printf(
            "Verification FAILED\n");
    }




    cudaFree(d_A);

    cudaFree(d_B);

    cudaFree(d_C);



    free(h_A);

    free(h_B);

    free(h_C);

    free(h_ref);



    cudaEventDestroy(start);

    cudaEventDestroy(stop);



    return 0;
}