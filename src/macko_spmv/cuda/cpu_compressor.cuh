#include <cuda_fp16.h>

#define COMPRESSOR_UPDIV(a, b) (a + b - 1) / b
#define COMPRESSOR_FOR(i, n) for (int i = 0; i < n; ++i)

void compress_rows_cpu(
    const __half *M, int M_rows, int M_cols,
    int **row_indices, unsigned char **deltas, __half **values,
    int *compressed_size, int *compressed_padded_size)
{

    *row_indices = (int *)calloc((M_rows + 1), sizeof(int));
    *values = (__half *)calloc((M_rows * M_cols + 8), sizeof(__half));
    *deltas = (unsigned char *)calloc((M_rows * M_cols + 8), sizeof(unsigned char));

    (*row_indices)[0] = 0;

    int num_values = 0;
    COMPRESSOR_FOR(row, M_rows)
    {
        unsigned int delta = 0;
        COMPRESSOR_FOR(col, M_cols)
        {
            float val = __half2float(M[row * M_cols + col]);
            if (val != 0.0 || delta == 15)
            {
                int delta_index = num_values / 2;
                int delta_subindex = num_values % 2;
                if (delta_subindex == 0)
                    (*deltas)[delta_index] = 0;
                (*deltas)[delta_index] += (delta) << (4 * delta_subindex);
                (*values)[num_values] = val;
                num_values += 1;
                delta = 0;
            }
            else
            {
                delta += 1;
            }
        }
        (*row_indices)[row + 1] = num_values;
    }
    (*compressed_size) = num_values;
    (*compressed_padded_size) = COMPRESSOR_UPDIV(num_values, 8) * 8;

    (*values) = (__half *)realloc(*values, (*compressed_padded_size) * sizeof(__half));
    (*deltas) = (unsigned char *)realloc(*deltas, ((*compressed_padded_size) / 2) * sizeof(unsigned char));
}


void compress_rows_int8_cpu(
    const int8_t *M, int M_rows, int M_cols,
    int **row_indices, unsigned char **deltas, int8_t **values,
    int *compressed_size, int *compressed_padded_size)
{

    *row_indices = (int *)calloc((M_rows + 1), sizeof(int));
    *values = (int8_t *)calloc((M_rows * M_cols + 8), sizeof(int8_t));
    *deltas = (unsigned char *)calloc((M_rows * M_cols + 8), sizeof(unsigned char));

    (*row_indices)[0] = 0;

    int num_values = 0;
    COMPRESSOR_FOR(row, M_rows)
    {
        unsigned int delta = 0;
        COMPRESSOR_FOR(col, M_cols)
        {
            int8_t val = (M[row * M_cols + col]);
            if (val != 0.0 || delta == 15)
            {
                int delta_index = num_values / 2;
                int delta_subindex = num_values % 2;
                if (delta_subindex == 0)
                    (*deltas)[delta_index] = 0;
                (*deltas)[delta_index] += (delta) << (4 * delta_subindex);
                (*values)[num_values] = val;
                num_values += 1;
                delta = 0;
            }
            else
            {
                delta += 1;
            }
        }
        (*row_indices)[row + 1] = num_values;
    }
    (*compressed_size) = num_values;
    (*compressed_padded_size) = COMPRESSOR_UPDIV(num_values, 8) * 8;

    (*values) = (int8_t *)realloc(*values, (*compressed_padded_size) * sizeof(int8_t));
    (*deltas) = (unsigned char *)realloc(*deltas, ((*compressed_padded_size) / 2) * sizeof(unsigned char));
}