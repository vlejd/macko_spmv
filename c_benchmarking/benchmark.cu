#include <iostream>
#include <vector>
#include <cublas_v2.h>
#include <cstdlib>
#include <stdlib.h>
#include <cassert>
#include <iomanip>
#include <cusparse_v2.h>
#include <chrono>
#include "../src/macko_spmv/cuda/kernels.cuh"
#include "utils.cuh"

using namespace std;

void run_mv(int kernel_id, TestData *test_data, cudaStream_t stream, cublasHandle_t handle)
{
    // 0 is for fp16 outputs
    // 1 is for fp32 outputs
    test_data->format_used = 1;
    switch (kernel_id)
    {
    case 0: // cublas fp32
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasCheck(cublasSgemv(
            handle,
            CUBLAS_OP_T,
            test_data->M_cols,
            test_data->M_rows,
            &alpha,
            test_data->M_float_d, test_data->M_cols,
            test_data->V_float_d, 1, &beta,
            test_data->C_float_d, 1));
        test_data->format_used = 0;
        break;
    }
    case 1: // cublas fp16 compute
    {
        __half alpha = 1.0f;
        __half beta = 0.0f;
        cublasCheck(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            test_data->M_rows,
            1,
            test_data->M_cols,
            &alpha,
            test_data->M_half_d, CUDA_R_16F, test_data->M_cols,
            test_data->V_half_d, CUDA_R_16F, test_data->M_cols,
            &beta,
            test_data->C_half_d, CUDA_R_16F, test_data->M_rows,
            CUBLAS_COMPUTE_16F,
            CUBLAS_GEMM_DEFAULT));
        break;
    }
    case 2: // cublas fp32 compute
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        cublasCheck(cublasGemmEx(
            handle,
            CUBLAS_OP_T, CUBLAS_OP_N,
            test_data->M_rows,
            1,
            test_data->M_cols,
            &alpha,
            test_data->M_half_d, CUDA_R_16F, test_data->M_cols,
            test_data->V_half_d, CUDA_R_16F, test_data->M_cols,
            &beta,
            test_data->C_half_d, CUDA_R_16F, test_data->M_rows,
            CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT));
        break;
    }
    case 3:
    {
        cusparseCheck(cusparseSpMM(test_data->sp_handle,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   CUSPARSE_OPERATION_NON_TRANSPOSE,
                                   &test_data->alpha_float,
                                   test_data->SpMatA,
                                   test_data->DnMatB,
                                   &test_data->beta_float,
                                   test_data->DnMatC,
                                   CUDA_R_32F,
                                   test_data->CuSparse_Algorithm,
                                   test_data->Buffer));
        break;
    }
    // TIDI case 4: // int8 times fp16!
    case 7:
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        dim3 grid_size(UPDIV(test_data->M_rows, BLOCK_SIZE));
        dim3 block_size(WARP_SIZE, BLOCK_SIZE);
        int mem_size = 0;
        macko_spmv<true><<<grid_size, block_size, mem_size, stream>>>(
            test_data->M_values_d, test_data->M_deltas_d, test_data->M_row_indices_d,
            test_data->V_half_d,
            test_data->M_rows, test_data->M_cols, alpha, beta,
            test_data->C_half_d);
        break;
    }
    case 8:
    {
        float alpha = 1.0f;
        float beta = 0.0f;
        dim3 grid_size(UPDIV(test_data->M_rows, BLOCK_SIZE));
        dim3 block_size(WARP_SIZE, BLOCK_SIZE);
        int mem_size = 0;
        macko_spmv_int8<true><<<grid_size, block_size, mem_size, stream>>>(
            test_data->M_int8_values_d, test_data->M_int8_deltas_d, test_data->M_int8_row_indices_d,
            test_data->V_half_d,
            test_data->M_rows, test_data->M_cols, alpha, beta,
            test_data->C_half_d);
        break;
    }

    default:
        cerr << "NO KERNEL" << endl;
        assert(false);
    }
}

int main(int argc, char **argv)
{
    int device = 0;
    cudaCheck(cudaSetDevice(device));
    cublasHandle_t handle;
    cublasCheck(cublasCreate(&handle));

    Params params(argc, argv);
    if (!params.ok)
        return 1;

    srand(params.seed);
    auto cache_flush = CacheFlush(device);
    cout << "Cache size: " << cache_flush.l2_cache_size << endl;

    vector<float> measured_times_us;
    measured_times_us.reserve(params.timing_reps);

    TestData test_data(params.bits_per_value, params.M_rows, params.M_cols, params.density);

    if (params.debug)
    {
        test_data.print_matrices();
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasCheck(cublasSetStream(handle, stream));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int algo_dtype_used;
    run_mv(params.kernel_id, &test_data, stream, handle);
    cudaCheck(cudaStreamSynchronize(stream));
    algo_dtype_used = test_data.format_used;
    test_data.move_output_to_host(algo_dtype_used);

    run_mv(0, &test_data, stream, handle);
    cudaCheck(cudaStreamSynchronize(stream));
    test_data.move_output_to_ref_host();
    if (params.debug)
    {
        test_data.print_output_and_ref(algo_dtype_used);
    }
    test_data.check_output_to_ref(algo_dtype_used);

    // warmup
    for (int warmup_rep = 0; warmup_rep < params.warmup_reps; warmup_rep++)
    {
        run_mv(params.kernel_id, &test_data, stream, handle);
    }

    // Time kernel launch time
    int kernel_start_reps = 100;
    double kernel_launch_time_us = 0.0;
    for (int kernel_start_rep = 0; kernel_start_rep < kernel_start_reps; kernel_start_rep++)
    {
        auto chrono_start = chrono::high_resolution_clock::now();
        run_mv(params.kernel_id, &test_data, stream, handle);
        auto chrono_end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(chrono_end - chrono_start);
        kernel_launch_time_us += double(duration.count());
    }
    kernel_launch_time_us = kernel_launch_time_us / kernel_start_reps;

    // Time kernel execution
    float ms = 0;
    for (int timing_rep = 0; timing_rep < params.timing_reps; timing_rep++)
    {
        if (params.flush_cache)
        {
            cache_flush.flush();
        }
        cudaCheck(cudaEventRecord(start, stream));
        run_mv(params.kernel_id, &test_data, stream, handle);
        cudaCheck(cudaEventRecord(stop, stream));
        cudaCheck(cudaEventSynchronize(stop));
        cudaCheck(cudaGetLastError());
        cudaEventElapsedTime(&ms, start, stop);
        measured_times_us.push_back(ms * 1000);
    }

    float times_min_us = vector_min(measured_times_us);
    float times_mean_us = vector_mean(measured_times_us);

    cout << "Kernel id: " << params.kernel_id << endl;
    cout << "Bits per value: " << params.bits_per_value << endl;
    cout << "Last time (us): " << ms * 1000 << endl;
    cout << "Max  time (us): " << vector_max(measured_times_us) << endl;
    cout << "Min  time (us): " << times_min_us << endl;
    cout << "Mean time (us): " << times_mean_us << endl;
    cout << "Mean kernel launch time (us): " << kernel_launch_time_us << endl;
    cout << "Dense Bandwidth best (Gbps): " << test_data.compute_dense_bandwidth_Gbps(times_min_us) << endl;
    cout << "Dense Bandwidth mean (Gbps): " << test_data.compute_dense_bandwidth_Gbps(times_mean_us) << endl;
    cout << "Dense flops best (Gflops): " << test_data.compute_dense_Gflops(times_min_us) << endl;
    cout << "Dense flops mean (Gflops): " << test_data.compute_dense_Gflops(times_mean_us) << endl;
    cout << "Sparse flops best (Gflops): " << test_data.compute_sparse_Gflops(times_min_us) << endl;
    cout << "Sparse flops mean (Gflops): " << test_data.compute_sparse_Gflops(times_mean_us) << endl;
    cout << "Effective density: " << test_data.compute_effective_density() << " " << test_data.density << endl;

    cout << endl;
    cout << "Final results:\t" << params.bits_per_value << "\t" << params.M_rows << "\t" << params.M_cols << "\t" << params.kernel_id << "\t" << params.density
         << "\t" << int(times_mean_us)
         << "\t" << int(test_data.compute_dense_Gflops(times_mean_us))
         << "\t" << int(test_data.compute_sparse_Gflops(times_mean_us))
         << endl;
}
