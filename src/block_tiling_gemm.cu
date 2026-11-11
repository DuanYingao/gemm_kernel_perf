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



// ============================================================
// Block Tiling GEMM Kernel
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
__global__ void block_tiling_gemm(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int M,
    const int N,
    const int K)
{
    // Shared Memory Tile
    __shared__ float As[BM][BK]; 
    __shared__ float Bs[BK][BN];


    // ========================================================
    // Thread mapping
    //
    // 一个 block 256 threads
    //
    // A tile loading:
    // 16 x 16 threads
    //
    // B tile loading:
    // 16 x 16 threads
    //
    // C compute:
    // 16 x 16 threads
    //
    // 每个 thread:
    // 
    // ========================================================


    // -------- Load A tile mapping --------
    constexpr int A_LOAD_X = 32;
    constexpr int A_LOAD_Y = BLOCK_SIZE / A_LOAD_X;


    const int a_thread_x = threadIdx.x % A_LOAD_X;
    const int a_thread_y = threadIdx.x / A_LOAD_X;



    // -------- Load B tile mapping --------
    constexpr int B_LOAD_X = 32;
    constexpr int B_LOAD_Y = BLOCK_SIZE / B_LOAD_X;


    const int b_thread_x = threadIdx.x % B_LOAD_X;
    const int b_thread_y = threadIdx.x / B_LOAD_X;



    // -------- Compute C mapping --------
    constexpr int C_THREAD_X = 16;
    constexpr int C_THREAD_Y = BLOCK_SIZE / C_THREAD_X;


    const int c_thread_x = threadIdx.x % C_THREAD_X;
    const int c_thread_y = threadIdx.x / 16 % C_THREAD_Y;



    // 每个线程负责的输出 tile
    constexpr int TM = BM / C_THREAD_Y;
    constexpr int TN = BN / C_THREAD_X;



    // Register tile
    float c_register[TM][TN] = {0.0f};

    // ========================================================
    // 运算过程
    // 1.先分块,一个block负责一个C矩阵的BMxBN大小的块，一共有1024 / BM * 1024 / BN 个block，也就是Grid里的block数量，这些block并行计算，用syncthreads()同步最终结果
    // 2.一个block沿k方向循环加载A和B的tile，并进行矩阵乘法计算
    // 3.由于一个block只有256个线程，无法直接用所有线程同时计算整个BMxBN的块，因此需要将BMxBN的块用for循环分解
    // 4.最外圈循环按BK的k一个个走，循环中A按一列读，B按一行读，先从N方向循环，一次循环读出256个元素，循环TN次，再从M方向下降c_thread_x的距离，再进行一次TN个行循环，重复TM次
    //
    // 
    // ========================================================

    // ========================================================
    // K 方向循环
    // ========================================================
    for (int k0 = 0; k0 < K; k0 += BK)
    {

        // ----------------------------------------------------
        // 存入 A tile
        // ----------------------------------------------------
        for (int i = a_thread_y;
             i < BM;
             i += A_LOAD_Y)
        {
            const int row = blockIdx.y * BM + i;
            const int col = k0 + a_thread_x;


            As[i][a_thread_x] =
                (row < M && col < K)
                    ? A[row * K + col]
                    : 0.0f;
        }



        // ----------------------------------------------------
        // 存入 B tile
        // ----------------------------------------------------
        for (int i = b_thread_y;
             i < BK;
             i += B_LOAD_Y)
        {
            for(int j = b_thread_x;
                j < BN;
                j += B_LOAD_X)
            {
                const int row = k0 + i;
                const int col = blockIdx.x * BN + j;

                Bs[i][j] =
                    (row < K && col < N)
                        ? B[row * N + col]
                        : 0.0f;
            }

        }



        __syncthreads();



        // ----------------------------------------------------
        // Compute:
        //
        // C += A_tile × B_tile
        //
        // 使用 register 累积结果
        // ----------------------------------------------------
        // #pragma unroll 展开加速，但寄存器代价大
        for (int k = 0; k < BK; ++k)
        {
            for (int i = 0; i < TM; ++i)
            {
                const int row =
                    c_thread_y +
                    i * C_THREAD_Y;


                for (int j = 0; j < TN; ++j)
                {
                    const int col =
                        c_thread_x +
                        j * C_THREAD_X;

                    // 读取AS时，一列32个全是lane0，全conflict
                    c_register[i][j] +=
                        As[row][k] * 
                        Bs[k][col];
                }
            }
        }


        __syncthreads();
    }



    // ========================================================
    // 写回 Global Memory
    // ========================================================
    for (int i = 0; i < TM; ++i)
    {
        const int row =
            blockIdx.y * BM +
            c_thread_y +
            i * C_THREAD_Y;


        for (int j = 0; j < TN; ++j)
        {
            const int col =
                blockIdx.x * BN +
                c_thread_x +
                j * C_THREAD_X;


            if (row < M && col < N)
            {
                C[row * N + col] =
                    c_register[i][j];
            }
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
    // int minGridSize, blockSize;
    // cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, block_tiling_gemm, 0, 0);
    // printf("Optimal block size: %d\n", blockSize);

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 32; // RTX 2060 每个 SM 48KB shared memory，128x16x4B=8KB，一个 block 2个 tiling 16KB，共享内存可以容纳 3 个block

    constexpr int BLOCK_SIZE = 256; // RTX 2060 每个 SM 4个 block，每个 block 8个 warp


    dim3 block(BLOCK_SIZE);

    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM);



    // -------------------------------
    // Warmup
    // -------------------------------
    block_tiling_gemm<
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


    block_tiling_gemm<
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