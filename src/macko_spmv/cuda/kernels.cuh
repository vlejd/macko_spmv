#include <cuda_runtime.h>
#include <cublas_v2.h>

#define BLOCK_SIZE 32
#define UPDIV(a, b) (a + b - 1) / b
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

template <typename T>
__inline__ __device__ T warpReduceSum(T val)
{
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
    {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

template <typename T>
__inline__ __device__ T warp_exclusive_sum(T val)
{
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    T tmp = val;
    for (int offset = 1; offset < WARP_SIZE; offset <<= 1)
    {
        T y = __shfl_up_sync(FULL_MASK, tmp, offset);
        if (lane >= offset)
            tmp += y;
    }
    return tmp - val;
}

#define LOAD_SIZE 8
template <bool reuse_value_from_c>
__global__ void macko_spmv(
    const __half *__restrict__ M_values, const unsigned char *__restrict__ M_deltas, const int *__restrict__ M_row_indices,
    const __half *__restrict__ V,
    const int M_rows, const int M_cols, const float alpha, const float beta,
    __half *__restrict__ C)
{
    int lane = threadIdx.x % 32;
    int m_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    int row_begin_ordinal = M_row_indices[m_row];
    int row_end_ordinal = M_row_indices[m_row + 1];

    int row_begin_ordinal_aligned = row_begin_ordinal - (row_begin_ordinal % LOAD_SIZE);
    int elements_from_previous_row = row_begin_ordinal % LOAD_SIZE;

    int sum_deltas_in_row = 0;
    float acc = 0;

    if (m_row >= M_rows)
    {
        return;
    }

    for (int warp_row_ordinal = row_begin_ordinal_aligned; warp_row_ordinal < row_end_ordinal; warp_row_ordinal += WARP_SIZE * LOAD_SIZE)
    {
        int thread_row_ordinal = warp_row_ordinal + lane * LOAD_SIZE;
        const float4 *M_values_vec = reinterpret_cast<const float4 *>(M_values);
        const unsigned int *M_deltas_vec = reinterpret_cast<const unsigned int *>(M_deltas);

        float4 m_values_raw = {0, 0, 0, 0};
        unsigned int m_deltas_raw = {0};
        if (thread_row_ordinal < row_end_ordinal)
        {
            m_values_raw = M_values_vec[thread_row_ordinal / LOAD_SIZE];
            m_deltas_raw = M_deltas_vec[thread_row_ordinal / LOAD_SIZE];
        }
        __half *m_values = reinterpret_cast<__half *>(&m_values_raw);

#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++)
        {
            if (thread_row_ordinal + i < row_begin_ordinal)
            {
                unsigned int delta_mask = ~(15 << (4 * i));
                m_deltas_raw &= delta_mask;
                m_values[i] = {0};
            }
        }

        int sum_deltas = 0;
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++)
        {
            sum_deltas += int((m_deltas_raw >> (4 * i)) & 15);
        }

        sum_deltas_in_row += warp_exclusive_sum(sum_deltas);
        int vals_to_skip = (thread_row_ordinal < row_begin_ordinal) ? elements_from_previous_row : 0;

#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++)
        {
            if (thread_row_ordinal + i >= vals_to_skip && thread_row_ordinal + i < row_end_ordinal)
            {
                sum_deltas_in_row += int((m_deltas_raw >> (4 * i)) & 15);
                int real_delta = sum_deltas_in_row + thread_row_ordinal + i - row_begin_ordinal;
                __half m_value = m_values[i];
                real_delta = (real_delta >= M_cols) ? M_cols - 1 : real_delta;
                real_delta = (real_delta <= 0) ? 0 : real_delta;
                __half v_value = V[real_delta];
                acc += __half2float(m_value) * __half2float(v_value);
            }
        }
        sum_deltas_in_row = __shfl_sync(0xffffffff, sum_deltas_in_row, 31);
    }
    acc = warpReduceSum(acc);
    if (lane == 0)
    {
        if constexpr (reuse_value_from_c){
            C[m_row] = __float2half(alpha * acc + beta * __half2float(C[m_row]));
        } else {
            C[m_row] = __float2half(alpha * acc);
        }
    }
}


// TODO Optimization needed
template <bool reuse_value_from_c>
__global__ void macko_spmv_int8(
    const int8_t *__restrict__ M_values, const unsigned char *__restrict__ M_deltas, const int *__restrict__ M_row_indices,
    const __half *__restrict__ V,
    const int M_rows, const int M_cols, const float alpha, const float beta,
    __half *__restrict__ C)
{
    // TODO, check precision
    int lane = threadIdx.x % 32;
    int m_row = blockIdx.x * BLOCK_SIZE + threadIdx.y;

    int row_begin_ordinal = M_row_indices[m_row];
    int row_end_ordinal = M_row_indices[m_row + 1];

    int row_begin_ordinal_aligned = row_begin_ordinal - (row_begin_ordinal % LOAD_SIZE);
    int elements_from_previous_row = row_begin_ordinal % LOAD_SIZE;

    int sum_deltas_in_row = 0;
    float acc = 0;

    if (m_row >= M_rows)
    {
        return;
    }

    for (int warp_row_ordinal = row_begin_ordinal_aligned; warp_row_ordinal < row_end_ordinal; warp_row_ordinal += WARP_SIZE * LOAD_SIZE)
    {
        int thread_row_ordinal = warp_row_ordinal + lane * LOAD_SIZE;
        const float2 *M_values_vec = reinterpret_cast<const float2 *>(M_values);
        const unsigned int *M_deltas_vec = reinterpret_cast<const unsigned int *>(M_deltas);

        float2 m_values_raw = {0, 0};
        unsigned int m_deltas_raw = {0};
        if (thread_row_ordinal < row_end_ordinal)
        {
            m_values_raw = M_values_vec[thread_row_ordinal / LOAD_SIZE];
            m_deltas_raw = M_deltas_vec[thread_row_ordinal / LOAD_SIZE];
        }
        int8_t *m_values = reinterpret_cast<int8_t *>(&m_values_raw);

#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++)
        {
            if (thread_row_ordinal + i < row_begin_ordinal)
            {
                unsigned int delta_mask = ~(15 << (4 * i));
                m_deltas_raw &= delta_mask;
                m_values[i] = {0};
            }
        }

        int sum_deltas = 0;
#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++)
        {
            sum_deltas += int((m_deltas_raw >> (4 * i)) & 15);
        }

        sum_deltas_in_row += warp_exclusive_sum(sum_deltas);
        int vals_to_skip = (thread_row_ordinal < row_begin_ordinal) ? elements_from_previous_row : 0;

#pragma unroll 8
        for (int i = 0; i < LOAD_SIZE; i++)
        {
            if (thread_row_ordinal + i >= vals_to_skip && thread_row_ordinal + i < row_end_ordinal)
            {
                sum_deltas_in_row += int((m_deltas_raw >> (4 * i)) & 15);
                int real_delta = sum_deltas_in_row + thread_row_ordinal + i - row_begin_ordinal;
                int8_t m_value = m_values[i];
                real_delta = (real_delta >= M_cols) ? M_cols - 1 : real_delta;
                real_delta = (real_delta <= 0) ? 0 : real_delta;
                __half v_value = V[real_delta];
                acc += float(m_value) * __half2float(v_value);
            }
        }
        sum_deltas_in_row = __shfl_sync(0xffffffff, sum_deltas_in_row, 31);
    }
    acc = warpReduceSum(acc);
    if (lane == 0)
    {
        if constexpr (reuse_value_from_c){
            C[m_row] = __float2half(alpha * acc + beta * __half2float(C[m_row]));
        } else {
            C[m_row] = __float2half(alpha * acc);
        }
    }
}