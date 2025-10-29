#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cstdlib>
#include <stdlib.h>
#include <cassert>
#include <cusparse_v2.h>
#include "../src/macko_spmv/cuda/cpu_compressor.cuh"

using namespace std;

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

class Params
{
public:
    int bits_per_value = 16;
    int M_rows = 1;
    int M_cols = 1024;
    int warmup_reps = 10;
    int timing_reps = 10;
    bool flush_cache = false;
    bool debug = false;
    int kernel_id = 0;
    int seed = 0;
    float density = 1.0;
    bool ok = true;

    Params(int argc, char **argv)
    {
        if (argc != 11)
        {
            ok = false;
            cerr << "Wrong ussage." << endl;
            cerr << "<bits_per_value> <rows> <cols> <warmup> <timing> <flush> <debug> <kernel_id> <seed> <density>" << endl;
            return;
        }

        try
        {
            bits_per_value = stoi(argv[1]);
            M_rows = stoi(argv[2]);
            M_cols = stoi(argv[3]);
            warmup_reps = stoi(argv[4]);
            timing_reps = stoi(argv[5]);
            flush_cache = stoi(argv[6]);
            debug = stoi(argv[7]);
            kernel_id = stoi(argv[8]);
            seed = stoi(argv[9]);
            density = stof(argv[10]);
        }
        catch (const exception &e)
        {
            ok = false;
            cerr << "Error: Invalid arguments: " << e.what() << endl;
            return;
        }
        cout << "Kernel id: " << kernel_id << endl;
        cout << "M_rows, M_cols, density: " << M_rows << " " << M_cols << " " << density << endl;
        cout << "Warmup reps / timing reps: " << warmup_reps << "/" << timing_reps << endl;
        cout << "Seed: " << seed << endl;
        cout << "Flush: " << flush_cache << endl;
        cout << endl;
    }
};

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

float vector_max(vector<float> v)
{
    assert(v.size());
    float ret = v[0];
    for (auto x : v)
    {
        ret = max(ret, x);
    }
    return ret;
}

float vector_min(vector<float> v)
{
    assert(v.size());
    float ret = v[0];
    for (auto x : v)
    {
        ret = min(ret, x);
    }
    return ret;
}

float vector_mean(vector<float> v)
{
    assert(v.size());
    float ret = 0;
    for (auto x : v)
    {
        ret += x;
    }
    return ret / v.size();
}

void print_matrix(float *M, int M_rows, int M_cols)
{
    int max_size = 64;
    cout << "[";
    for (int r = 0; r < M_rows; r++)
    {
        cout << "[";
        for (int c = 0; c < M_cols && c < max_size; c++)
        {
            cout << M[r * M_cols + c] << ", ";
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

void print_matrix(int *M, int M_rows, int M_cols)
{
    int max_size = 64;
    cout << "[";
    for (int r = 0; r < M_rows; r++)
    {
        cout << "[";
        for (int c = 0; c < M_cols && c < max_size; c++)
        {
            cout << M[r * M_cols + c] << ", ";
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

void print_matrix(unsigned char *M, int M_rows, int M_cols)
{
    int max_size = 256;
    cout << "[";
    for (int r = 0; r < M_rows; r++)
    {
        cout << "[";
        for (int c = 0; c < M_cols && c < max_size; c += 2)
        {
            cout << (int(M[r * M_cols + c]) & 15) << ", ";
            cout << (int(M[r * M_cols + c]) >> 4) << ", ";
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

void print_matrix(int8_t *M, int M_rows, int M_cols)
{
    int max_size = 64;
    cout << "[";
    for (int r = 0; r < M_rows; r++)
    {
        cout << "[";
        for (int c = 0; c < M_cols && c < max_size; c++)
        {
            cout << int(M[r * M_cols + c]) << ", ";
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

class TestData
{
public:
    int bits_per_value = 16;
    int M_rows, M_cols;
    int BATCH = 1;  // Maybe will be relevant later
    int format_used;

    float density;
    float *M_float_h, *V_float_h, *C_float_h, *C_float_ref_h;
    float *M_float_d, *V_float_d, *C_float_d;

    __half *M_half_h, *V_half_h, *C_half_h, *C_half_ref_h;
    __half *M_half_d, *V_half_d, *C_half_d;

    int8_t *M_int8_h, *M_int8_d;

    int M_float_size;
    int V_float_size;
    int M_half_size;
    int V_half_size;
    int M_int8_size;

    // MACKO compression
    unsigned char *M_deltas_h, *M_deltas_d;
    __half *M_values_h, *M_values_d;
    int *M_row_indices_h, *M_row_indices_d;
    int M_sparse_size;
    int M_sparse_padded_size;

    // MACKO compression int8
    unsigned char *M_int8_deltas_h, *M_int8_deltas_d;
    int8_t *M_int8_values_h, *M_int8_values_d;
    int *M_int8_row_indices_h, *M_int8_row_indices_d;
    int M_int8_sparse_size;
    int M_int8_sparse_padded_size;


    // CUBLAS constants
    const float alpha_float = 1.0;
    const float beta_float = 0.0;

    // Cusparse thnings
    void *Buffer = NULL;
    size_t bufferSize = 0;
    int *csrRowPtr;
    int *csrColInd;
    half *csrVal;
    cusparseHandle_t sp_handle = 0;
    cusparseSpMatDescr_t SpMatA;
    cusparseDnMatDescr_t DnMatA, DnMatB, DnMatC;
    int64_t numRowTMP, numColTMP, NNZ_1;
    cusparseSpMMAlg_t CuSparse_Algorithm;

    TestData(int arg_bits_per_value, int arg_M_rows, int arg_M_cols, float arg_density)
    {
        bits_per_value = arg_bits_per_value;
        M_rows = arg_M_rows;
        M_cols = arg_M_cols;
        density = arg_density;

        // float stuff
        M_float_size = UPDIV(M_cols * M_rows, 4) * 4;
        V_float_size = UPDIV(M_cols, 4) * 4;
        M_float_h = (float *)malloc(M_float_size * sizeof(float));
        V_float_h = (float *)malloc(V_float_size * sizeof(float));
        C_float_h = (float *)malloc(M_rows * sizeof(float));
        C_float_ref_h = (float *)malloc(M_rows * sizeof(float));
        // Allocate cuda memory
        cudaCheck(cudaMalloc((void **)&M_float_d, M_float_size * sizeof(float)));
        cudaCheck(cudaMalloc((void **)&V_float_d, V_float_size * sizeof(float)));
        cudaCheck(cudaMalloc((void **)&C_float_d, M_rows * sizeof(float)));

        // __half stuff
        M_half_size = UPDIV(M_cols * M_rows, 8) * 8;
        V_half_size = UPDIV(M_cols, 8) * 8;
        M_half_h = (__half *)malloc(M_half_size * sizeof(__half));
        V_half_h = (__half *)malloc(V_half_size * sizeof(__half));
        C_half_h = (__half *)malloc(M_rows * sizeof(__half));
        C_half_ref_h = (__half *)malloc(M_rows * sizeof(__half));
        // Allocate cuda memory
        cudaCheck(cudaMalloc((void **)&M_half_d, M_half_size * sizeof(__half)));
        cudaCheck(cudaMalloc((void **)&V_half_d, V_half_size * sizeof(__half)));
        cudaCheck(cudaMalloc((void **)&C_half_d, M_rows * sizeof(__half)));

        // int8 stuff
        M_int8_size = UPDIV(M_cols*M_rows, 8)*8;
        M_int8_h = (int8_t *)malloc(M_int8_size* sizeof(int8_t));
        cudaCheck(cudaMalloc((void **)&M_int8_d, M_int8_size * sizeof(int8_t)));

        init_non_zero_matrix(M_float_h, M_rows * M_cols);
        init_non_zero_vector(V_float_h, M_cols);

        sparsify();
        convert_to_halfs();
        
        compress();

        convert_to_int8();
        compress_int8();

        // Copy to cuda
        cudaCheck(cudaMemcpy(M_float_d, M_float_h, M_float_size * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(V_float_d, V_float_h, V_float_size * sizeof(float), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(M_half_d, M_half_h, M_half_size * sizeof(__half), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(V_half_d, V_half_h, V_half_size * sizeof(__half), cudaMemcpyHostToDevice));
        cudaCheck(cudaMemcpy(M_int8_d, M_int8_h, M_int8_size * sizeof(int8_t), cudaMemcpyHostToDevice));

        setupCusparse();
    }

    void setupCusparse()
    {
        cudaMalloc(&csrRowPtr, sizeof(int) * (M_rows + 1));
        CuSparse_Algorithm = CUSPARSE_SPMM_CSR_ALG1;
        cusparseCreate(&sp_handle);
        cusparseSetStream(sp_handle, 0);
        // Create Dense Matrix
        cusparseCheck(cusparseCreateDnMat(&DnMatA,
                                          M_rows,
                                          M_cols,
                                          M_cols,
                                          M_half_d,
                                          CUDA_R_16F,
                                          CUSPARSE_ORDER_ROW));
        cusparseCheck(cusparseCreateDnMat(&DnMatB, M_cols, BATCH, M_cols, V_half_d, CUDA_R_16F, CUSPARSE_ORDER_COL));
        cusparseCheck(
            cusparseCreateDnMat(&DnMatC, M_rows, BATCH, M_rows, C_half_d, CUDA_R_16F, CUSPARSE_ORDER_COL));

        cusparseCheck(cusparseCreateCsr(&SpMatA,
                                        M_rows,
                                        M_cols,
                                        0,
                                        csrRowPtr,
                                        NULL,
                                        NULL,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_32I,
                                        CUSPARSE_INDEX_BASE_ZERO,
                                        CUDA_R_16F));

        cusparseCheck(
            cusparseDenseToSparse_bufferSize(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bufferSize));
        cudaMalloc(&Buffer, bufferSize);
        cusparseCheck(
            cusparseDenseToSparse_analysis(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer));

        cusparseCheck(cusparseSpMatGetSize(SpMatA, &numRowTMP, &numColTMP, &NNZ_1));

        cudaMalloc(&csrColInd, NNZ_1 * sizeof(int));
        cudaMalloc(&csrVal, NNZ_1 * sizeof(half));

        cusparseCheck(cusparseCsrSetPointers(SpMatA, csrRowPtr, csrColInd, csrVal));
        cusparseCheck(cusparseDenseToSparse_convert(sp_handle, DnMatA, SpMatA, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, Buffer));

        cusparseCheck(cusparseSpMM_bufferSize(sp_handle,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                                              &alpha_float,
                                              SpMatA,
                                              DnMatB,
                                              &beta_float,
                                              DnMatC,
                                              CUDA_R_32F,
                                              CuSparse_Algorithm,
                                              &bufferSize));
        cudaFree(Buffer);
        cudaMalloc(&Buffer, bufferSize);
    }

    void init_non_zero_matrix(float *M, int size)
    {
        for (int i = 0; i < size; i++)
        {
            do{
                if(bits_per_value==16){
                    M[i] = float(rand() % 101 - 50) / 100;
                } else if (bits_per_value==8)
                {
                    M[i] = float(rand() % 101 - 50);
                } else{
                    assert(false);
                }
            } while(M[i]==0);
        }
    }

    void init_non_zero_vector(float *M, int size)
    {
        for (int i = 0; i < size; i++)
        {
            do{
                M[i] = float(rand() % 101-50) / 100;
            } while(M[i]==0);
        }
    }

    void sparsify()
    {
        // compute number of elements
        int size = M_cols * M_rows;
        int nnz = int(density * float(size));
        int zeros = size - nnz;

        // sample mask & set values
        int ones_to_sample = nnz;
        int zeros_to_sample = zeros;

        for (int row = 0; row < M_rows; row++)
        {
            for (int col = 0; col < M_cols; col++)
            {
                bool is_one = (rand() % (ones_to_sample + zeros_to_sample)) < ones_to_sample;
                // Make the value zero
                if (!is_one)
                {
                    M_float_h[row * M_cols + col] = 0;
                    zeros_to_sample--;
                }
                else
                {
                    ones_to_sample--;
                }
            }
        }
    }

    void compress(){
        compress_rows_cpu(
            M_half_h, M_rows, M_cols, &M_row_indices_h, &M_deltas_h, &M_values_h, &M_sparse_size, &M_sparse_padded_size);

        cudaCheck(cudaMalloc((void **)&M_row_indices_d, (M_rows + 1) * sizeof(int)));
        cudaCheck(cudaMemcpy(M_row_indices_d, M_row_indices_h, (M_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc((void **)&M_values_d, (M_sparse_padded_size) * sizeof(__half)));
        cudaCheck(cudaMemcpy(M_values_d, &(*M_values_h), (M_sparse_padded_size) * sizeof(__half), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc((void **)&M_deltas_d, (M_sparse_padded_size / 2) * sizeof(unsigned char)));
        cudaCheck(cudaMemcpy(M_deltas_d, &(*M_deltas_h), (M_sparse_padded_size / 2) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }

    void compress_int8(){
        compress_rows_int8_cpu(
            M_int8_h, M_rows, M_cols, &M_int8_row_indices_h, &M_int8_deltas_h, &M_int8_values_h, &M_int8_sparse_size, &M_int8_sparse_padded_size);

        cudaCheck(cudaMalloc((void **)&M_int8_row_indices_d, (M_rows + 1) * sizeof(int)));
        cudaCheck(cudaMemcpy(M_int8_row_indices_d, M_int8_row_indices_h, (M_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc((void **)&M_int8_values_d, (M_int8_sparse_padded_size) * sizeof(int8_t)));
        cudaCheck(cudaMemcpy(M_int8_values_d, &(*M_int8_values_h), (M_int8_sparse_padded_size) * sizeof(int8_t), cudaMemcpyHostToDevice));

        cudaCheck(cudaMalloc((void **)&M_int8_deltas_d, (M_int8_sparse_padded_size / 2) * sizeof(unsigned char)));
        cudaCheck(cudaMemcpy(M_int8_deltas_d, &(*M_int8_deltas_h), (M_int8_sparse_padded_size / 2) * sizeof(unsigned char), cudaMemcpyHostToDevice));
    }

    void convert_to_halfs()
    {
        for (int i = 0; i < M_rows * M_cols; i++)
        {
            M_half_h[i] = __float2half(M_float_h[i]);
        }
        for (int i = 0; i < M_cols; i++)
        {
            V_half_h[i] = __float2half(V_float_h[i]);
        }
    }

    void convert_to_int8()
    {
        for (int i = 0; i < M_rows * M_cols; i++)
        {
            //M_float_h should have int values.
            M_int8_h[i] = int8_t(M_float_h[i]);
        }
    }

    void print_matrices()
    {
        cout << "V float:\n";
        print_matrix(V_float_h, 1, M_cols);

        cout << "M float:\n";
        print_matrix(M_float_h, M_rows, M_cols);

        cout << "V half:\n";
        print_matrix(V_half_h, 1, M_cols);

        cout << "M half:\n";
        print_matrix(M_half_h, M_rows, M_cols);

        cout << "M int8:\n";
        print_matrix(M_int8_h, M_rows, M_cols);

        cout << "M values: " << M_sparse_size << " " << M_sparse_padded_size << endl;
        print_matrix(M_values_h, 1, M_sparse_size);

        cout << "M deltas:\n";
        print_matrix(M_deltas_h, 1, M_sparse_size);

        cout << "M indices:\n";
        print_matrix(M_row_indices_h, 1, M_rows + 1);

        cout << "M int8 values: " << M_int8_sparse_size << " " << M_int8_sparse_padded_size << endl;
        print_matrix(M_int8_values_h, 1, M_int8_sparse_size);

        cout << "M int8 deltas:\n";
        print_matrix(M_int8_deltas_h, 1, M_int8_sparse_size);

        cout << "M int8 indices:\n";
        print_matrix(M_int8_row_indices_h, 1, M_rows + 1);
    }

    void move_output_to_host(int algo_dtype_used)
    {
        if (algo_dtype_used == 0)
        {
            cout << "Float" << endl;
            cudaCheck(cudaMemcpy(C_float_h, C_float_d, M_rows * sizeof(float), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemset((void *)C_float_d, 0, M_rows * sizeof(float)));
        }
        else if (algo_dtype_used == 1)
        {
            cout << "Half" << endl;
            cudaCheck(cudaMemcpy(C_half_h, C_half_d, M_rows * sizeof(__half), cudaMemcpyDeviceToHost));
            cudaCheck(cudaMemset((void *)C_half_d, 0, M_rows * sizeof(__half)));
        }
    }

    void move_output_to_ref_host()
    {
        cudaCheck(cudaMemcpy(C_float_ref_h, C_float_d, M_rows * sizeof(float), cudaMemcpyDeviceToHost));
        cudaCheck(cudaMemset((void *)C_float_d, 0, M_rows * sizeof(float)));
    }

    float abs(float x)
    {
        return (x > 0) ? x : -x;
    }

    void check_output_to_ref(int algo_dtype_used)
    {
        float maxdif = 0.;
        for (int i = 0; i < M_rows; i++)
        {
            float ref_val = C_float_ref_h[i];
            float found_val;
            if (algo_dtype_used == 0)
            {
                found_val = C_float_h[i];
            }
            else if (algo_dtype_used == 1)
            {
                found_val = __half2float(C_half_h[i]);
            }
            maxdif = max(maxdif, this->abs(found_val - ref_val));
        }
        cout << "Max diff: " << maxdif << endl;
        if (maxdif > 0.1)
        {
            cout << "@@@@@@ LOOKS BIG @@@@@@@@" << endl;
        }
        cout << endl;
    }

    float compute_dense_Gflops(float us)
    {
        return 2. * M_rows * M_cols / (us * 1e3);
    }

    float compute_sparse_Gflops(float us)
    {
        return 2. * M_rows * M_cols * density / (us * 1e3);
    }

    float compute_dense_bandwidth_Gbps(float us)
    {
        float bytes = sizeof(__half) * (M_rows * M_cols + M_rows + M_cols);
        float bandwidth = (bytes / (us * 1000));
        return bandwidth;
    }

    float compute_sparse_bandwidth_Gbps(float us)
    {
        float bytes = sizeof(__half) * (M_rows * M_cols * density + M_rows + M_cols);
        float bandwidth = (bytes / (us * 1000));
        return bandwidth;
    }

    float compute_effective_density()
    {
        double dense_bytes = M_cols * M_rows * 2;
        double compresses_bytes = 2.5 * M_sparse_padded_size + M_cols * 4;
        return compresses_bytes / dense_bytes;
    }

    void print_output_and_ref(int algo_dtype_used)
    {
        cout << "C " << endl;
        if (algo_dtype_used == 0)
        {
            cout << "C_float_h";
            print_matrix(C_float_h, 1, M_rows);
        }
        else if (algo_dtype_used == 1)
        {
            cout << "C_half_h";
            print_matrix(C_half_h, 1, M_rows);
        }
        cout << "C ref " << endl;
        print_matrix(C_float_ref_h, 1, M_rows);
    }

    ~TestData()
    {
        free(M_float_h);
        free(V_float_h);
        free(C_float_h);
        free(C_float_ref_h);
        cudaFree(M_float_d);
        cudaFree(V_float_d);
        cudaFree(C_float_d);

        free(M_half_h);
        free(V_half_h);
        free(C_half_h);
        free(C_half_ref_h);
        cudaFree(M_half_d);
        cudaFree(V_half_d);
        cudaFree(C_half_d);
        free(M_row_indices_h);

        cudaFree(M_deltas_d);
        cudaFree(M_values_d);
        cudaFree(M_row_indices_d);
    }
};
