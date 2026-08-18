#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s:%d: %s\\n", __FILE__, __LINE__, \
                         cudaGetErrorString(err__));                            \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// A 的第二维加 1：原来的 As[BM][32] 使同一 warp 读不同 A 行时
// 命中同一个 bank（4-way conflict）。BK + 1 让相邻行错开一个 bank。
template <int BM, int BK>
__device__ inline void load_tile_A(const float* __restrict__ A,
                                   float As[BM][BK + 1], int block_row, int bk,
                                   int tid, int M, int K) {
    constexpr int TILE_SIZE = BM * BK;
    const bool in_bounds = (block_row + 1) * BM <= M && bk + BK <= K;

    // BK=16、BM=128 的专用协作加载映射。
    //
    // As 的行跨度为 17，以避免计算阶段的 shared-load conflict。若保持线性
    // idx=tid 映射，一个 warp 会同时写两条相邻行，因 17 mod 32=17 而发生
    // shared-store conflict。这里让半个 warp 写 row r，另半个 warp 写 row r+16：
    // 两组 bank 分别为 [0,15] 与 [16,31]（经 r 偏移后仍不重叠）。
    // 两个半 warp 的 global load 各是连续 16 个 float，仍完全合并。
    if constexpr (BM == 128 && BK == 16) {
        const int warp = tid >> 5;
        const int lane = tid & 31;
        const int col = lane & 15;
        const int half_warp = lane >> 4;

        if (in_bounds) {
            // 不展开：展开 8 次会同时保留 8 组地址/载入值，使寄存器数从约 74
            // 增至 100，并令 SM 只能驻留 2 个 block。这里优先保持 3 block/SM。
            #pragma unroll 1
            for (int iter = 0; iter < 8; ++iter) {
                const int pair = iter * 8 + warp;      // [0, 63]
                const int row = (pair & 15) + (pair >> 4) * 32 + half_warp * 16;
                As[row][col] = A[(block_row * BM + row) * K + bk + col];
            }
        } else {
            #pragma unroll 1
            for (int iter = 0; iter < 8; ++iter) {
                const int pair = iter * 8 + warp;
                const int row = (pair & 15) + (pair >> 4) * 32 + half_warp * 16;
                const int global_row = block_row * BM + row;
                const int global_col = bk + col;
                As[row][col] = (global_row < M && global_col < K)
                                   ? A[global_row * K + global_col]
                                   : 0.0f;
            }
        }
        return;
    }

    if (in_bounds) {
        // 对 BK=16，/ 和 % 会编译为移位与掩码；同一 warp 连续读 global memory。
        #pragma unroll
        for (int idx = tid; idx < TILE_SIZE; idx += blockDim.x) {
            const int row = idx / BK;
            const int col = idx % BK;
            As[row][col] = A[(block_row * BM + row) * K + bk + col];
        }
    } else {
        for (int idx = tid; idx < TILE_SIZE; idx += blockDim.x) {
            const int row = idx / BK;
            const int col = idx % BK;
            const int global_row = block_row * BM + row;
            const int global_col = bk + col;
            As[row][col] = (global_row < M && global_col < K)
                               ? A[global_row * K + global_col]
                               : 0.0f;
        }
    }
}

template <int BK, int BN>
__device__ inline void load_tile_B(const float* __restrict__ B,
                                   float Bs[BK][BN], int block_col, int bk,
                                   int tid, int K, int N) {
    constexpr int TILE_SIZE = BK * BN;
    const bool in_bounds = bk + BK <= K && (block_col + 1) * BN <= N;

    if (in_bounds) {
        #pragma unroll
        for (int idx = tid; idx < TILE_SIZE; idx += blockDim.x) {
            const int row = idx / BN;
            const int col = idx % BN;
            Bs[row][col] = B[(bk + row) * N + block_col * BN + col];
        }
    } else {
        for (int idx = tid; idx < TILE_SIZE; idx += blockDim.x) {
            const int row = idx / BN;
            const int col = idx % BN;
            const int global_row = bk + row;
            const int global_col = block_col * BN + col;
            Bs[row][col] = (global_row < K && global_col < N)
                               ? B[global_row * N + global_col]
                               : 0.0f;
        }
    }
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void thread_tiling_gemm_optimized(
    const float* __restrict__ A, const float* __restrict__ B,
    float* __restrict__ C, int M, int N, int K) {
    static_assert(BM / TM == 16 && BN / TN == 16,
                  "The 4x2-warp, 4x8-lane mapping covers a 16x16 thread grid");

    __shared__ float As[BM][BK + 1];  // +1 prevents shared-memory bank conflicts.
    __shared__ float Bs[BK][BN];

    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;
    const int warp_row = warp_id >> 1;  // 4 x 2 warps in one block.
    const int warp_col = warp_id & 1;
    const int lane_row = lane_id >> 3;  // 4 x 8 threads within one warp.
    const int lane_col = lane_id & 7;
    const int thread_row = (warp_row * 4 + lane_row) * TM;
    const int thread_col = (warp_col * 8 + lane_col) * TN;
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;

    float reg_c[TM][TN] = {0.0f};
    float reg_a[TM];
    float reg_b[TN];

    for (int bk = 0; bk < K; bk += BK) {
        load_tile_A<BM, BK>(A, As, block_row, bk, tid, M, K);
        load_tile_B<BK, BN>(B, Bs, block_col, bk, tid, K, N);
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int i = 0; i < TM; ++i)
                reg_a[i] = As[thread_row + i][k];
            #pragma unroll
            for (int j = 0; j < TN; ++j)
                reg_b[j] = Bs[k][thread_col + j];

            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j)
                    reg_c[i][j] = fmaf(reg_a[i], reg_b[j], reg_c[i][j]);
            }
        }
        __syncthreads();
    }

    const int global_row = block_row * BM + thread_row;
    const int global_col = block_col * BN + thread_col;

    // TN=4 时把四个连续结果作为一个 float4 写回。原实现的逐 float 写回会使
    // 相邻 lane 的同一条 store 相隔 16 B，正是报告中 “8/32 bytes per sector” 的来源。
    if (global_row + TM <= M && global_col + TN <= N && (N & 3) == 0) {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            *reinterpret_cast<float4*>(C + (global_row + i) * N + global_col) =
                make_float4(reg_c[i][0], reg_c[i][1], reg_c[i][2], reg_c[i][3]);
        }
    } else {
        #pragma unroll
        for (int i = 0; i < TM; ++i) {
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                if (global_row + i < M && global_col + j < N)
                    C[(global_row + i) * N + global_col + j] = reg_c[i][j];
            }
        }
    }
}

void fill_random(float* data, int size) {
    for (int i = 0; i < size; ++i)
        data[i] = static_cast<float>(std::rand()) / RAND_MAX;
}

void sgemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = sum;
        }
}

bool verify_result(const float* gpu, const float* cpu, int size) {
    for (int i = 0; i < size; ++i) {
        if (std::fabs(gpu[i] - cpu[i]) > 1e-3f) {
            std::printf("Mismatch %d: GPU=%f CPU=%f\\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    constexpr int M = 1024, N = 1024, K = 1024;
    constexpr int BM = 128, BN = 64;
    // BK=16 使静态共享内存约为 12.8 KiB；在 RTX 2060 上可维持 3 个 block/SM。
    // load_tile_A 内的专用映射消除了该配置原本的 shared-store conflict。
    constexpr int BK = 16;
    constexpr int TM = 8, TN = 4;

    const size_t size_A = static_cast<size_t>(M) * K * sizeof(float);
    const size_t size_B = static_cast<size_t>(K) * N * sizeof(float);
    const size_t size_C = static_cast<size_t>(M) * N * sizeof(float);
    float *h_A = static_cast<float*>(std::malloc(size_A));
    float *h_B = static_cast<float*>(std::malloc(size_B));
    float *h_C = static_cast<float*>(std::malloc(size_C));
    float *h_ref = static_cast<float*>(std::malloc(size_C));
    fill_random(h_A, M * K);
    fill_random(h_B, K * N);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size_A));
    CUDA_CHECK(cudaMalloc(&d_B, size_B));
    CUDA_CHECK(cudaMalloc(&d_C, size_C));
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice));

    const dim3 block(256);
    const dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    thread_tiling_gemm_optimized<BM, BN, BK, TM, TN>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    constexpr int RUNS = 50;
    CUDA_CHECK(cudaEventRecord(start));
    for (int run = 0; run < RUNS; ++run)
        thread_tiling_gemm_optimized<BM, BN, BK, TM, TN>
            <<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const float avg_ms = total_ms / RUNS;
    const double gflops = 2.0 * M * N * K / (avg_ms * 1e6);
    std::printf("optimized: %.3f ms, %.2f GFLOP/s\\n", avg_ms, gflops);

    CUDA_CHECK(cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost));
    sgemm_cpu(h_A, h_B, h_ref, M, N, K);
    std::printf("Verification %s\\n", verify_result(h_C, h_ref, M * N) ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_A)); CUDA_CHECK(cudaFree(d_B)); CUDA_CHECK(cudaFree(d_C));
    std::free(h_A); std::free(h_B); std::free(h_C); std::free(h_ref);
    CUDA_CHECK(cudaEventDestroy(start)); CUDA_CHECK(cudaEventDestroy(stop));
}
