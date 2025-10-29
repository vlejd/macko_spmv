#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cstdlib>
#include <stdlib.h>
#include <cassert>
#include <iomanip>
#include <cusparse_v2.h>
#include <numeric>
#include <cmath>

using namespace std;

#define for_i(i, n) for (int i = 0; i < int(n); ++i)
#define UPDIV(a, b) (a + b - 1) / b
#define cublasCheck(status) (cublasCheckF(status, __FILE__, __LINE__))

void cublasCheckF(cublasStatus_t status, const char *file, int line)
{
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        printf("[CUBLAS ERROR] at file %s:%d:\n%d\n", file, line,
               status);
        exit(EXIT_FAILURE);
    }
}

#define cudaCheck(err) (cudaCheckF(err, __FILE__, __LINE__))

void cudaCheckF(cudaError_t error, const char *file, int line)
{
    if (error != cudaSuccess)
    {
        printf("[CUDA ERROR] at file %s:%d:\n%s\n", file, line,
               cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
};

#define cusparseCheck(status) (cusparseCheckF(status, __FILE__, __LINE__))

void cusparseCheckF(cusparseStatus_t status, const char *file, int line)
{
    if (status != CUSPARSE_STATUS_SUCCESS)
    {
        printf("[CUSPARSE ERROR] at file %s:%d:\n%d\n%s\n", file, line,
               status, cusparseGetErrorString(status));
        exit(EXIT_FAILURE);
    }
}

class CacheFlush
{
public:
    int l2_cache_size;
    size_t cache_flush_data_size;
    int8_t *cache_flush_data_d;

    CacheFlush(int device)
    {
        l2_cache_size = 0;
        cudaCheck(cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device));
        cache_flush_data_size = l2_cache_size * 2;
        cudaCheck(cudaMalloc((void **)&cache_flush_data_d, cache_flush_data_size));
        cudaCheck(cudaDeviceSynchronize());
    }

    ~CacheFlush()
    {
        cudaFree(cache_flush_data_d);
    }

    void flush()
    {
        cudaCheck(cudaMemset((void *)cache_flush_data_d, 0, cache_flush_data_size));
        cudaCheck(cudaDeviceSynchronize());
        cudaCheck(cudaGetLastError());
    }
};

__half *malloc_half(int size)
{
    return (__half *)malloc(size * sizeof(__half));
}

__half *cuda_malloc_half(int size)
{
    size = UPDIV(size, 8) * 8;
    __half *ptr;
    cudaCheck(cudaMalloc((void **)&ptr, size * sizeof(__half)));
    return ptr;
}

__half *cuda_malloc_half_from_host(__half *host, int size)
{
    __half *ptr = cuda_malloc_half(size);
    cudaCheck(cudaMemcpy(ptr, host, size * sizeof(__half), cudaMemcpyHostToDevice));
    return ptr;
}

void init_matrix(__half *M, int size)
{
    for (int i = 0; i < size; i++)
    {
        M[i] = __float2half(float(rand() % 11 - 5) / 10);
    }
}

void transpose(__half *src, __half *dst, int src_rows, int src_cols)
{
    for_i(row, src_rows)
    {
        for_i(col, src_cols)
        {
            dst[col * src_rows + row] = src[row * src_cols + col];
        }
    }
}

void print_matrix(__half *M, int M_rows, int M_cols)
{
    int max_size = 64;
    cout << "[";
    for (int r = 0; r < M_rows; r++)
    {
        cout << "[";
        for (int c = 0; c < M_cols && c < max_size; c++)
        {
            cout << __half2float(M[r * M_cols + c]) << ", ";
            if (c == max_size - 1)
            {
                cout << "... \n";
                break;
            }
        }
        cout << "],\n";
        if (r == max_size - 1)
        {
            cout << "... \n";
            break;
        }
    }
    cout << "]\n";
}

class Data
{
public:
    int A_rows;
    int A_cols_B_rows;
    int B_cols;
    int seed;
    int debug;
    int cache_flush;
    int kerne_id;
    __half *A_h, *At_h, *B_h, *Bt_h;
    __half *A_d, *At_d, *B_d, *Bt_d;
    __half *C_h, *C_d, *Ct_h, *Ct_d;
    __half alpha = 1.0f;
    __half beta = 0.0f;

    int device = 0;
    cublasHandle_t handle;

    Data(int argc, char **argv)
    {
        A_rows = stoi(argv[1]);
        A_cols_B_rows = stoi(argv[2]);
        B_cols = stoi(argv[3]);
        seed = stoi(argv[4]);
        debug = stoi(argv[5]);
        cache_flush = stoi(argv[6]);
        kerne_id = stoi(argv[7]);

        srand(seed);
        A_h = malloc_half(A_rows * A_cols_B_rows);
        B_h = malloc_half(A_cols_B_rows * B_cols);
        At_h = malloc_half(A_rows * A_cols_B_rows);
        Bt_h = malloc_half(A_cols_B_rows * B_cols);
        C_h = malloc_half(A_rows * B_cols);
        Ct_h = malloc_half(A_rows * B_cols);

        cudaCheck(cudaSetDevice(device));
        cublasCheck(cublasCreate(&handle));

        // Init matrices
        init_matrix(A_h, A_rows * A_cols_B_rows);
        init_matrix(B_h, A_cols_B_rows * B_cols);

        // Transpose them
        transpose(A_h, At_h, A_rows, A_cols_B_rows);
        transpose(B_h, Bt_h, A_cols_B_rows, B_cols);

        // Copy to cuda
        A_d = cuda_malloc_half_from_host(A_h, A_rows * A_cols_B_rows);
        At_d = cuda_malloc_half_from_host(At_h, A_rows * A_cols_B_rows);
        B_d = cuda_malloc_half_from_host(B_h, A_cols_B_rows * B_cols);
        Bt_d = cuda_malloc_half_from_host(Bt_h, A_cols_B_rows * B_cols);

        C_d = cuda_malloc_half(A_rows * B_cols);
        Ct_d = cuda_malloc_half(A_rows * B_cols);
    }

    void print()
    {
        if (debug)
        {
            cout << "A" << endl;
            print_matrix(A_h, A_rows, A_cols_B_rows);
            cout << "B" << endl;
            print_matrix(B_h, A_cols_B_rows, B_cols);

            cout << "At" << endl;
            print_matrix(At_h, A_cols_B_rows, A_rows);
            cout << "Bt" << endl;
            print_matrix(Bt_h, B_cols, A_cols_B_rows);
        }
    }

    void print_out()
    {
        if (debug)
        {
            cout << "C" << endl;
            print_matrix(C_h, A_rows, B_cols);

            cout << "Ct" << endl;
            print_matrix(Ct_h, B_cols, A_rows);
        }
    }

    void move_output_to_host()
    {
        cudaCheck(cudaMemcpy(C_h, C_d, A_rows * B_cols * sizeof(__half), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemset((void *)C_d, 0, A_rows * B_cols * sizeof(__half)));
        cudaCheck(cudaMemcpy(Ct_h, Ct_d, A_rows * B_cols * sizeof(__half), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemset((void *)Ct_d, 0, A_rows * B_cols * sizeof(__half)));
    }
};

#define TTT 111
#define TTN 110
#define TNT 101
#define TNN 100
#define NTT 11
#define NTN 10
#define NNT 1
#define NNN 0

void run_kerne(int kernel_id, Data &data)
{
    switch (kernel_id)
    {
    case TTT:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            data.A_rows,
            data.B_cols,
            data.A_cols_B_rows,
            &data.alpha,
            data.At_d, CUDA_R_16F, data.A_rows,
            data.Bt_d, CUDA_R_16F, data.A_cols_B_rows,
            &data.beta, data.Ct_d, CUDA_R_16F, data.A_rows,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    case TNT:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            data.A_rows,
            data.B_cols,
            data.A_cols_B_rows,
            &data.alpha,
            data.At_d, CUDA_R_16F, data.A_rows,
            data.B_d, CUDA_R_16F, data.B_cols,
            &data.beta, data.Ct_d, CUDA_R_16F, data.A_rows,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    case NTT:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            data.A_rows,
            data.B_cols,
            data.A_cols_B_rows,
            &data.alpha,
            data.A_d, CUDA_R_16F, data.A_cols_B_rows,
            data.Bt_d, CUDA_R_16F, data.A_cols_B_rows,
            &data.beta, data.Ct_d, CUDA_R_16F, data.A_rows,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    case NNT:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            data.A_rows,
            data.B_cols,
            data.A_cols_B_rows,
            &data.alpha,
            data.A_d, CUDA_R_16F, data.A_cols_B_rows,
            data.B_d, CUDA_R_16F, data.B_cols,
            &data.beta, data.Ct_d, CUDA_R_16F, data.A_rows,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    case NNN:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            data.B_cols,
            data.A_rows,
            data.A_cols_B_rows,
            &data.alpha,
            data.B_d, CUDA_R_16F, data.B_cols,
            data.A_d, CUDA_R_16F, data.A_cols_B_rows,
            &data.beta, data.C_d, CUDA_R_16F, data.B_cols,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    case NTN:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            data.B_cols,
            data.A_rows,
            data.A_cols_B_rows,
            &data.alpha,
            data.Bt_d, CUDA_R_16F, data.A_cols_B_rows,
            data.A_d, CUDA_R_16F, data.A_cols_B_rows,
            &data.beta, data.C_d, CUDA_R_16F, data.B_cols,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    case TTN:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_T, CUBLAS_OP_T,
            data.B_cols,
            data.A_rows,
            data.A_cols_B_rows,
            &data.alpha,
            data.Bt_d, CUDA_R_16F, data.A_cols_B_rows,
            data.At_d, CUDA_R_16F, data.A_rows,
            &data.beta, data.C_d, CUDA_R_16F, data.B_cols,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    case TNN:
    {

        cublasCheck(cublasGemmEx(
            data.handle,
            CUBLAS_OP_N, CUBLAS_OP_T,
            data.B_cols,
            data.A_rows,
            data.A_cols_B_rows,
            &data.alpha,
            data.B_d, CUDA_R_16F, data.B_cols,
            data.At_d, CUDA_R_16F, data.A_rows,
            &data.beta, data.C_d, CUDA_R_16F, data.B_cols,
            CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT));
        break;
    }
    default:
    {
        cerr << "NO KERNEL" << endl;
        assert(false);
    }
    }
}

int main(int argc, char **argv)
{
    auto data = Data(argc, argv);
    auto cache_flush = CacheFlush(data.device);
    cout << "Cache size: " << cache_flush.l2_cache_size << endl;

    run_kerne(data.kerne_id, data);
    data.move_output_to_host();

    data.print_out();

    int runs = 1000;

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    vector<cudaEvent_t> starts(runs);
    vector<cudaEvent_t> ends(runs);
    for_i(i, runs)
    {
        cudaEventCreate(&starts[i]);
        cudaEventCreate(&ends[i]);
    }

    // warmup
    for_i(i, 10)
    {
        run_kerne(data.kerne_id, data);
    }
    cudaCheck(cudaStreamSynchronize(stream));

    // timing
    for_i(i, runs)
    {
        if (data.cache_flush)
        {
            cache_flush.flush();
        }
        cudaCheck(cudaEventRecord(starts[i], stream));
        run_kerne(data.kerne_id, data);
        cudaCheck(cudaEventRecord(ends[i], stream));
    }

    float runtime_ms = 0;
    vector<float> measured_times_us;
    for_i(i, runs)
    {
        cudaCheck(cudaEventSynchronize(ends[i]));
        cudaCheck(cudaGetLastError());
        cudaEventElapsedTime(&runtime_ms, starts[i], ends[i]);
        measured_times_us.push_back(runtime_ms * 1000);
    }
    cout << measured_times_us.size() << endl;
    cout << measured_times_us[0] << endl;
    float mean_time_us = accumulate(measured_times_us.begin(), measured_times_us.end(), 0.0f) / runs;
    mean_time_us = ceil(mean_time_us);
    double std_time_us = 0.0;
    for_i(i, runs)
    {
        float diff = measured_times_us[i] - mean_time_us;
        std_time_us += diff * diff;
    }
    std_time_us = sqrt(std_time_us / runs);

    double bytes_to_move = (sizeof(__half) * ((double)(data.A_rows) * (double)(data.A_cols_B_rows) +
                                              (double)(data.A_cols_B_rows) * (double)(data.B_cols) +
                                              (double)(data.A_rows) * (double)(data.B_cols)));
    double bandwidth_gbps = bytes_to_move / ((double)(mean_time_us) * 1e3);
    double flops_to_do = 2.0 * ((double)(data.A_rows) * (double)(data.A_cols_B_rows) * (double)(data.B_cols));
    double gflops = flops_to_do / ((double)(mean_time_us) * 1e3);

    // float mean_time_us = 0.0;
    cout << "Kernel: " << data.kerne_id << endl;
    cout << "Problem size A_rows X A_cols_B_rows X B_cols: " << data.A_rows << " X " << data.A_cols_B_rows << " X " << data.B_cols << endl;
    cout << "Average time us: " << mean_time_us << endl;
    cout << "Std of time us: " << ceil(std_time_us * 100) / 100 << endl;
    cout << "Bandwidth gbps: " << bandwidth_gbps << endl;
    cout << "Gflops: " << gflops << endl;
}