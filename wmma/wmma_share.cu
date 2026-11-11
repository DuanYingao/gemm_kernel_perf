#include "common.h"

#define BLOCK_THREAD_ROWS 64
#define BLOCK_THREAD_COLS 4
#define BLOCK_SIZE 256 

#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])
constexpr int NUM_WARPS = 8;
using namespace nvcuda;

template <
    const size_t BLOCK_ROWS,
    const size_t BLOCK_COLS,
    const size_t WARP_TILES_M,
    const size_t WARP_TILES_N
>
__global__ void wmma_share(
    const half* __restrict__ A,
    const half* __restrict__ B,
    half* __restrict__ C,
    size_t M,
    size_t N,
    size_t K)
{
    // This implementation uses a fixed 8-warp configuration.
   static_assert(BLOCK_ROWS == 128, "BLOCK_ROWS must be 128");
   static_assert(BLOCK_COLS == 128, "BLOCK_COLS must be 128");
   static_assert(WARP_TILES_M == 2, "WARP_TILES_M must be 2");
   static_assert(WARP_TILES_N == 4, "WARP_TILES_N must be 4");

    constexpr size_t SMEM_PAD = 8;

    constexpr size_t A_SMEM_STRIDE = WMMA_K + SMEM_PAD; 
    constexpr size_t B_SMEM_STRIDE = BLOCK_COLS + SMEM_PAD;

    // Shared-memory tiles
    __shared__ half s_a[BLOCK_ROWS][A_SMEM_STRIDE];
    __shared__ half s_b[WMMA_K][B_SMEM_STRIDE];

    // Temporary storage for converting FP32 accumulators to FP16.
    __shared__ float s_c[NUM_WARPS][WMMA_M * WMMA_N];

    const size_t tid = threadIdx.x;
    const size_t lane_id = tid & (WARP_SIZE - 1);

    const size_t warp_id = tid / WARP_SIZE;

    // 4 x 2 warp arrangement
    const size_t warp_m = warp_id / 2;
    const size_t warp_n = warp_id % 2;

    // --------------------------------------------------------
    // Cooperative loading indices
    // --------------------------------------------------------

    // A: 128 x 16
    // Two threads load 8 half values each per row.
    const size_t a_row = tid / 2;
    const size_t a_col = (tid % 2) * 8;

    // B: 16 x 128
    // Sixteen threads load 8 half values each per row.
    const size_t b_row = tid / 16;
    const size_t b_col = (tid % 16) * 8;

    const size_t K_tiles = K / WMMA_K;

    // --------------------------------------------------------
    // Accumulator fragments
    // --------------------------------------------------------

    wmma::fragment<
        wmma::accumulator,
        WMMA_M, WMMA_N, WMMA_K,
        float
    > c_frag[WARP_TILES_M][WARP_TILES_N];

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i) {
        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {
            wmma::fill_fragment(c_frag[i][j], 0.0f);
        }
    }

    // --------------------------------------------------------
    // Main K loop
    // --------------------------------------------------------

    for (size_t k = 0; k < K_tiles; ++k) {

        // Global-memory coordinates
        const size_t gmem_a_row =
            blockIdx.y * BLOCK_ROWS + a_row;

        const size_t gmem_b_col =
            blockIdx.x * BLOCK_COLS + b_col;

        const size_t gmem_a_idx =
            gmem_a_row * K + k * WMMA_K + a_col;

        const size_t gmem_b_idx =
            (k * WMMA_K + b_row) * N + gmem_b_col;

        // ----------------------------------------------------
        // Load A and B using 128-bit vectorized transactions
        // ----------------------------------------------------

        *reinterpret_cast<uint4*>(
            &s_a[a_row][a_col]
        ) = *reinterpret_cast<const uint4*>(
            &A[gmem_a_idx]
        );

        *reinterpret_cast<uint4*>(
            &s_b[b_row][b_col]
        ) = *reinterpret_cast<const uint4*>(
            &B[gmem_b_idx]
        );

        __syncthreads();

        // ----------------------------------------------------
        // Load WMMA fragments from shared memory
        // ----------------------------------------------------

        wmma::fragment<
            wmma::matrix_a,
            WMMA_M, WMMA_N, WMMA_K,
            half,
            wmma::row_major
        > a_frag[WARP_TILES_M];

        wmma::fragment<
            wmma::matrix_b,
            WMMA_M, WMMA_N, WMMA_K,
            half,
            wmma::row_major
        > b_frag[WARP_TILES_N];

        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; ++i) {

            const size_t smem_a_row =
                warp_m * WMMA_M * WARP_TILES_M
                + i * WMMA_M;

            wmma::load_matrix_sync(
                a_frag[i],
                &s_a[smem_a_row][0],
                A_SMEM_STRIDE
            );
        }

        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {

            const size_t smem_b_col =
                warp_n * WMMA_N * WARP_TILES_N
                + j * WMMA_N;

            wmma::load_matrix_sync(
                b_frag[j],
                &s_b[0][smem_b_col],
                B_SMEM_STRIDE
            );
        }

        // ----------------------------------------------------
        // Tensor Core matrix multiplication
        // ----------------------------------------------------

        #pragma unroll
        for (int i = 0; i < WARP_TILES_M; ++i) {

            #pragma unroll
            for (int j = 0; j < WARP_TILES_N; ++j) {

                wmma::mma_sync(
                    c_frag[i][j],
                    a_frag[i],
                    b_frag[j],
                    c_frag[i][j]
                );
            }
        }

        // Ensure all warps finish reading shared memory
        // before the next iteration overwrites it.
        __syncthreads();
    }

    // --------------------------------------------------------
    // Store FP32 accumulators and convert to FP16
    // --------------------------------------------------------

    #pragma unroll
    for (int i = 0; i < WARP_TILES_M; ++i) {

        #pragma unroll
        for (int j = 0; j < WARP_TILES_N; ++j) {

            // Each warp stores its fragment into its own
            // 16 x 16 shared-memory buffer.

            wmma::store_matrix_sync(
                &s_c[warp_id][0],
                c_frag[i][j],
                WMMA_N,
                wmma::mem_row_major
            );

            __syncthreads();

            // Output coordinates
            const size_t c_row_base =
                blockIdx.y * BLOCK_ROWS
                + warp_m * WMMA_M * WARP_TILES_M
                + i * WMMA_M;

            const size_t c_col_base =
                blockIdx.x * BLOCK_COLS
                + warp_n * WMMA_N * WARP_TILES_N
                + j * WMMA_N;

            // Each warp converts 256 FP32 values to FP16.
            for (int e = lane_id; e < WMMA_M * WMMA_N;
                 e += WARP_SIZE) {

                const size_t row = c_row_base + e / WMMA_N;
                const size_t col = c_col_base + e % WMMA_N;

                C[row * N + col] =
                    __float2half_rn(s_c[warp_id][e]);
            }

            // Prevent overwriting s_c before all warps finish.
            __syncthreads();
        }
    }
}


int main(){
    constexpr int M = 1024;
    constexpr int N = 1024;
    constexpr int K = 1024;

    const size_t size_A = M * K * sizeof(half);
    const size_t size_B = K * N * sizeof(half);
    const size_t size_C = M * N * sizeof(half);


    // -------------------------------
    // Host 内存分配
    // -------------------------------
    half* h_A = static_cast<half*>(malloc(size_A));
    half* h_B = static_cast<half*>(malloc(size_B));
    half* h_C = static_cast<half*>(malloc(size_C));


    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }


    fill_random(h_A, M * K);
    fill_random(h_B, K * N);



    // -------------------------------
    // Device 内存分配
    // -------------------------------
    half* d_A = nullptr;
    half* d_B = nullptr;
    half* d_C = nullptr;

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
    
   const size_t block_rows = 128;
   const size_t block_cols = 128;
   dim3 block(BLOCK_SIZE);
   dim3 grid(div_ceil(N, block_cols), div_ceil(M, block_rows));
   
    

    cudaEvent_t start;
    cudaEvent_t stop;


    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));


    CUDA_CHECK(cudaEventRecord(start));



    wmma_share<block_rows, block_cols, 2, 4><<<grid, block>>>(
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
    // 性能统计
    // -------------------------------
    const double flops =
        2.0 *
        static_cast<double>(M) *
        static_cast<double>(N) *
        static_cast<double>(K);


    const double gflops =
        flops /
        (elapsed_ms * 1e6);


    printf("--------------------------------\n");
    printf("Matrix size : %d x %d x %d\n",
           M,
           N,
           K);

    printf("Kernel time : %.3f ms\n",
           elapsed_ms);

    printf("Performance : %.2f GFLOPS\n",
           gflops);

    printf("--------------------------------\n");



    // -------------------------------
    // 拷贝结果回 Host
    // -------------------------------
    CUDA_CHECK(cudaMemcpy(
        h_C,
        d_C,
        size_C,
        cudaMemcpyDeviceToHost));



    // -------------------------------
    // 资源释放
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