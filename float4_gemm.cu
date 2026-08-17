#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <assert.h>

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
// Vectorized memory access
//
// 一次读取连续 4 个 float
//
// 要求:
// 地址 16 byte 对齐
// ============================================================

template<typename T>
__device__ inline float4 load_float4(const T* ptr)
{
    return *reinterpret_cast<const float4*>(ptr);
}



// ============================================================
// Warp-level Thread Tiling SGEMM
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

template <
    int BM,
    int BN,
    int BK,
    int TM,
    int TN
>
__global__ void float4_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M,
    int N,
    int K)
{


// ============================================================
// Shared Memory Swizzle Layout
//
// 第二维增加 padding
//
// 避免:
// thread0 -> bank0
// thread1 -> bank0
//
// ============================================================

constexpr int SWIZZLE = 4;


__shared__ float As[BM][BK + SWIZZLE];

__shared__ float Bs[BK][BN];



    // ========================================================
    // Thread / Warp ID
    // ========================================================
    const int tid =
        threadIdx.x;


    const int warp_id =
        tid >> 5;


    const int lane_id =
        tid & 31;



    // ========================================================
    // Warp Mapping
    //
    // Block:
    //
    // 8 warps
    //
    // warp layout:
    //
    // warp0 warp1
    // warp2 warp3
    // warp4 warp5
    // warp6 warp7
    //
    // ========================================================

    constexpr int WARP_N = 2;


    const int warp_row =
        warp_id / WARP_N;


    const int warp_col =
        warp_id % WARP_N;



    // ========================================================
    // Lane Mapping
    //
    // 一个 warp:
    //
    // 4 × 8 layout
    //
    // lane:
    //
    // row: 0~3
    // col: 0~7
    //
    // ========================================================

    const int lane_row =
        lane_id >> 3;


    const int lane_col =
        lane_id & 7;



    // ========================================================
    // Thread Tile Position
    //
    // 每个 thread:
    //
    // TM × TN
    //
    // ========================================================

    const int thread_row =
        (warp_row * 4 + lane_row)
        * TM;


    const int thread_col =
        (warp_col * 8 + lane_col)
        * TN;



    // ========================================================
    // Register Tile
    // ========================================================
    float reg_a[TM];

    float reg_b[TN];


    float reg_c[TM][TN] =
        {0.0f};



    const int block_row =
        blockIdx.y * BM;


    const int block_col =
        blockIdx.x * BN;



    // ========================================================
    // K Loop
    // ========================================================
    for(int k0 = 0;
        k0 < K;
        k0 += BK)
    {


        // ----------------------------------------------------
        // Load A Tile
        // ----------------------------------------------------

        // ========================================================
        // Vectorized Load A
        //
        // Global Memory:
        // 连续读取 float4
        //
        // Shared Memory:
        // swizzle 存储
        //
        // ========================================================


        for(int idx = tid * 4;
            idx < BM * BK;
            idx += blockDim.x * 4)
        {

            int row =
                idx / BK;


            int col =
                idx % BK;



            // 保证 float4 对齐
            assert(col % 4 == 0);



            int global_row =
                block_row + row;


            int global_col =
                k0 + col;



            if(global_row < M &&
            global_col + 3 < K)
            {

                float4 value =
                    load_float4(
                        &A[global_row*K + global_col]
                    );


                // swizzle 写入

                As[row][(col+0)^((row&3)<<2)] =
                    value.x;


                As[row][(col+1)^((row&3)<<2)] =
                    value.y;


                As[row][(col+2)^((row&3)<<2)] =
                    value.z;


                As[row][(col+3)^((row&3)<<2)] =
                    value.w;

            }
            else
            {

                for(int i=0;i<4;i++)
                {

                    if(global_row<M &&
                    global_col+i<K)
                    {
                        As[row][
                            (col+i)^((row&3)<<2)
                        ] =
                            A[global_row*K+
                            global_col+i];
                    }
                    else
                    {
                        As[row][
                            (col+i)^((row&3)<<2)
                        ]=0.0f;
                    }
                }
            }
        }



        // ----------------------------------------------------
        // Load B Tile
        // ----------------------------------------------------

        for(int idx = tid*4;
        idx < BK*BN;
        idx += blockDim.x*4)
    {

        int row =
            idx / BN;


        int col =
            idx % BN;


        int global_row =
            k0+row;


        int global_col =
            block_col+col;



        if(global_row<K &&
        global_col+3<N)
        {

            float4 value =
                load_float4(
                    &B[global_row*N+
                    global_col]
                );


            Bs[row][col+0]=value.x;

            Bs[row][col+1]=value.y;

            Bs[row][col+2]=value.z;

            Bs[row][col+3]=value.w;

        }
    }



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
                // ========================================================
                // Shared Memory Read
                //
                // 使用相同 swizzle 规则反解
                // ========================================================

                int smem_row =
                    thread_row+i;


                int smem_col =
                    k ^ ((smem_row & 3)<<2);



                reg_a[i] =
                    As[
                    smem_row
                    ][
                    smem_col
                    ];
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

            int row =
                block_row +
                thread_row +
                i;


            int col =
                block_col +
                thread_col +
                j;



            if(row<M &&
               col<N)
            {
                C[row*N+col] =
                    reg_c[i][j];
            }

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
    constexpr int BN=128;
    constexpr int BK=32;


    constexpr int TM=8;
    constexpr int TN=8;



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
        "Float4 GEMM\n");

    printf(
        "BM=%d BN=%d BK=%d TM=%d TN=%d\n",
        BM,BN,BK,TM,TN);



    // warmup

    float4_gemm<
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


        float4_gemm<
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