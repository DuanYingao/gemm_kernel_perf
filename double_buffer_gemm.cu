#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <iostream>


// ============================================================
// CUDA 错误检查
// ============================================================
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr,                                                  \
                    "CUDA Error [%s:%d]: %s\n",                              \
                    __FILE__,                                                \
                    __LINE__,                                                \
                    cudaGetErrorString(err));                                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)


#define FLOAT4(ptr) (reinterpret_cast<float4*>(&(ptr))[0])


// ============================================================
// Float4 Tiling GEMM Kernel
//
// 使用:
// 1. Shared Memory 缓存 A/B tile
// 2. Register 保存线程计算结果
// 3. 每个 block 计算 BM×BN 输出矩阵块
//
// C = A × B
//
// A: M × K
// B: K × N
// C: M × N
//
// Tile:
// A: BM × BK
// B: BK × BN
// C: BM × BN
// ============================================================
template <
    int BM,
    int BN,
    int BK,
    int BLOCK_SIZE
>
__device__ void load_global_to_shared(
    float* __restrict__ A,
    float* __restrict__ B,
    float (&As)[BK][BM],
    float (&Bs)[BK][BN],
    int M, int N, int K, int k0)
{
        // -------- Load A tile mapping --------
        // 为了让float4能案列读取读tileA，对As进行转置
        // constexpr int A_LOAD_Y = BLOCK_SIZE / A_LOAD_X;
        constexpr int A_LOAD_Y = BLOCK_SIZE >> 3;


        // const int a_thread_x = threadIdx.x % A_LOAD_X;
        // const int a_thread_y = threadIdx.x / A_LOAD_X;
        const int a_thread_x = threadIdx.x & 7;
        const int a_thread_y = threadIdx.x >> 3;



        // -------- Load B tile mapping --------
        // constexpr int B_LOAD_X = 16;
        // constexpr int B_LOAD_Y = BLOCK_SIZE / B_LOAD_X;
        constexpr int B_LOAD_Y = BLOCK_SIZE >> 4;

        // const int b_thread_x = threadIdx.x % B_LOAD_X;
        // const int b_thread_y = threadIdx.x / B_LOAD_X;
        const int b_thread_x = threadIdx.x & 15;
        const int b_thread_y = threadIdx.x >> 4;


        int a_j = a_thread_x * 4;
        
        for (int i = a_thread_y;
             i < BM;
             i += A_LOAD_Y)
        {
            const int row = blockIdx.y * BM + i;
            const int col = k0 + a_j;


            float4 a_val = FLOAT4(A[row * K + col]);

            As[a_j + 0][i] = a_val.x;

            As[a_j + 1][i] = a_val.y;

            As[a_j + 2][i] = a_val.z;

            As[a_j + 3][i] = a_val.w;
        }

        int b_j = b_thread_x * 4;

        for (int i = b_thread_y;
             i < BK;
             i += B_LOAD_Y)
        {
        
            const int row = k0 + i;
            const int col = blockIdx.x * BN + b_j;

            float4 b_val = FLOAT4(B[row * N + col]);

            Bs[i][b_j + 0] = b_val.x;

            Bs[i][b_j + 1] = b_val.y;

            Bs[i][b_j + 2] = b_val.z;

            Bs[i][b_j + 3] = b_val.w;
            
        }
}


template <
    int BM,
    int BN,
    int BK,
    int BLOCK_SIZE
>
__global__ void double_buffer_gemm(
    float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K)
{
    // Shared Memory Tile
    // 为了让float4能案列读取读tileA，对As进行转置
    __shared__ float As[2][BK][BM]; 
    __shared__ float Bs[2][BK][BN];

    // ========================================================
    // Thread mapping
    //
    // 一个 block 256 threads
    //
    // A tile loading:
    // 32 x 8 threads
    //
    // B tile loading:
    // 8 x 32 threads
    //
    // C compute:
    // 16 x 16 threads
    //
    // 每个 thread:
    // 
    // ========================================================


    // -------- Compute C mapping --------
    // constexpr int C_THREAD_X = 16;
    // constexpr int C_THREAD_Y = BLOCK_SIZE / C_THREAD_X;
    // constexpr int C_THREAD_Y = BLOCK_SIZE >> 4;


    // const int c_thread_x = threadIdx.x % C_THREAD_X;
    // const int c_thread_y = threadIdx.x / C_THREAD_X;
    const int c_thread_x = threadIdx.x & 15;
    const int c_thread_y = threadIdx.x >> 4;



    // 每个线程负责的输出 tile
    // constexpr int TM = BM / C_THREAD_Y;
    // constexpr int TN = BN / C_THREAD_X;
    constexpr int TM = BM >> 4;
    constexpr int TN = BN >> 4;



    // Register tile
    float4 a_frag_vec[2], b_frag_vec[2];
    float c_register[TM][TN] = {0.0f};

    int buf_idx = 0;
    load_global_to_shared<
        BM,
        BN,
        BK,
        BLOCK_SIZE>
        (A, B, As[0], Bs[0], M, N, K, 0);
    __syncthreads();

    // 加载数据到寄存器
    // const int row = c_thread_y * 4;
    
    a_frag_vec[0] = FLOAT4(As[0][0][c_thread_y * 4]);
    

    // const int col = c_thread_x * 4;
    
    b_frag_vec[0] = FLOAT4(Bs[0][0][c_thread_x * 4]);
    // ========================================================
    // K 方向循环
    // ========================================================
    for (int k0 = 0; k0 < K; k0 += BK)
    {

        // ----------------------------------------------------
        // 先存global到shared，再存shared到寄存器
        // 一个k，存下一个k的数据到寄存器，再计算当前k，一个循环一存一算，循环完BK后，写回C
        // 如果是第一个k，直接存下一个buffer的global到shared
        // ----------------------------------------------------
        int next_buf = 1 - buf_idx;


        for (int k = 0; k < BK; ++k)
        {
            

            if (k + 1 < BK) {
            
            a_frag_vec[1 - k % 2] = FLOAT4(As[buf_idx][k + 1][c_thread_y * 4]);
            
            b_frag_vec[1 - k % 2] = FLOAT4(Bs[buf_idx][k + 1][c_thread_x * 4]);

            }

            // 在第一个 k 步时，发射下一 tile 的 Global Memory 加载
            if (k == 0 && k0 < K) {
                load_global_to_shared<
        BM,
        BN,
        BK,
        BLOCK_SIZE>
        (A, B, As[next_buf], Bs[next_buf], M, N, K, k0 + BK);
            }


            // c_register变为4*4的连续区域，而不是跨块分布的16个独立数据

            // 第一行
            c_register[0][0] += a_frag_vec[k % 2].x * b_frag_vec[k % 2].x;

            c_register[0][1] += a_frag_vec[k % 2].x * b_frag_vec[k % 2].y;

            c_register[0][2] += a_frag_vec[k % 2].x * b_frag_vec[k % 2].z;

            c_register[0][3] += a_frag_vec[k % 2].x * b_frag_vec[k % 2].w;

            // 第二行
            
            c_register[1][0] += a_frag_vec[k % 2].y * b_frag_vec[k % 2].x;

            c_register[1][1] += a_frag_vec[k % 2].y * b_frag_vec[k % 2].y;

            c_register[1][2] += a_frag_vec[k % 2].y * b_frag_vec[k % 2].z;

            c_register[1][3] += a_frag_vec[k % 2].y * b_frag_vec[k % 2].w;

            // 第三行
            
            c_register[2][0] += a_frag_vec[k % 2].z * b_frag_vec[k % 2].x;

            c_register[2][1] += a_frag_vec[k % 2].z * b_frag_vec[k % 2].y;

            c_register[2][2] += a_frag_vec[k % 2].z * b_frag_vec[k % 2].z;

            c_register[2][3] += a_frag_vec[k % 2].z * b_frag_vec[k % 2].w;

            // 第四行

            c_register[3][0] += a_frag_vec[k % 2].w * b_frag_vec[k % 2].x;

            c_register[3][1] += a_frag_vec[k % 2].w * b_frag_vec[k % 2].y;

            c_register[3][2] += a_frag_vec[k % 2].w * b_frag_vec[k % 2].z;

            c_register[3][3] += a_frag_vec[k % 2].w * b_frag_vec[k % 2].w;
              
            
        }


        __syncthreads();

        buf_idx = next_buf;

        a_frag_vec[0] = FLOAT4(As[buf_idx][0][c_thread_y * 4]);
        
        b_frag_vec[0] = FLOAT4(Bs[buf_idx][0][c_thread_x * 4]);
    }



    // ========================================================
    // 写回 Global Memory
    // ========================================================

    for (int i = 0; i < TM; ++i)
    {
        const int row =
            blockIdx.y * BM +
            c_thread_y * 4 + i;

        const int col =
            blockIdx.x * BN +
            c_thread_x * 4;

        float4 c_vec = make_float4(c_register[i][0], c_register[i][1], 
                                    c_register[i][2], c_register[i][3]);
        if (row < M && col < N)
        {
            __stcg(reinterpret_cast<float4*>(&C[row * N + col]), c_vec);
        }
    }
}



// ============================================================
// 初始化矩阵
// ============================================================
void fill_random(float* data, const int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] =
            static_cast<float>(rand()) /
            static_cast<float>(RAND_MAX);
    }
}



// ============================================================
// Main
// ============================================================
int main()
{
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;


    const size_t size_A =
        M * K * sizeof(float);

    const size_t size_B =
        K * N * sizeof(float);

    const size_t size_C =
        M * N * sizeof(float);



    // -------------------------------
    // Host memory
    // -------------------------------
    float* h_A =
        static_cast<float*>(malloc(size_A));

    float* h_B =
        static_cast<float*>(malloc(size_B));

    float* h_C =
        static_cast<float*>(malloc(size_C));


    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr,
                "Host allocation failed\n");

        return EXIT_FAILURE;
    }


    fill_random(h_A, M * K);
    fill_random(h_B, K * N);



    // -------------------------------
    // Device memory
    // -------------------------------
    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;


    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));


    CUDA_CHECK(cudaMemcpy(
        d_A,
        h_A,
        size_A,
        cudaMemcpyHostToDevice));


    CUDA_CHECK(cudaMemcpy(
        d_B,
        h_B,
        size_B,
        cudaMemcpyHostToDevice));



    // -------------------------------
    // Kernel configuration
    // -------------------------------
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32; // RTX 2060 每个 SM 48KB shared memory，64x16x4B=4KB，一个 block 2个 tiling 8KB，共享内存可以容纳 4 个block

    
    constexpr int BLOCK_SIZE = 256; // RTX 2060 每个 SM 4个 block，每个 block 8个 warp


    dim3 block(BLOCK_SIZE);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);



    // -------------------------------
    // Warmup
    // -------------------------------
    double_buffer_gemm<
        BM,
        BN,
        BK,
        BLOCK_SIZE>
        <<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K);


    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());



    // -------------------------------
    // Benchmark
    // -------------------------------
    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));


    double_buffer_gemm<
        BM,
        BN,
        BK,
        BLOCK_SIZE>
        <<<grid, block>>>(
            d_A,
            d_B,
            d_C,
            M,
            N,
            K);


    CUDA_CHECK(cudaGetLastError());


    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaEventSynchronize(stop));



    float elapsed_ms = 0.0f;


    CUDA_CHECK(cudaEventElapsedTime(
        &elapsed_ms,
        start,
        stop));



    // -------------------------------
    // Performance
    // -------------------------------
    const double flops =
        2.0 *
        M *
        N *
        K;


    const double gflops =
        flops /
        (elapsed_ms * 1e6);



    printf("============================\n");
    printf("Matrix : %d x %d x %d\n",
           M,
           N,
           K);

    printf("Time   : %.3f ms\n",
           elapsed_ms);

    printf("GFLOPS : %.2f\n",
           gflops);

    printf("============================\n");



    // -------------------------------
    // Copy result
    // -------------------------------
    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        size_C,
        cudaMemcpyDeviceToHost));



    // -------------------------------
    // Release resources
    // -------------------------------
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));


    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));


    free(h_A);
    free(h_B);
    free(h_C);


    return 0;
}