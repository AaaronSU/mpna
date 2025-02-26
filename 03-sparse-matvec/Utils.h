//
// Simple class for a sparse matrix in CSR format

#ifndef UTILS_H
#define UTILS_H


template<typename IdType, typename ScalarType>
CSRMatrix<IdType, ScalarType> read_from_file(IdType size, ScalarType)
{
    IdType n = size * size;
    int nnz = 5 * n - 4 * size; // Number of non-zero elements

    CSRMatrix<IdType, ScalarType> laplacian(n, nnz);

    int idx = 0;
    for (IdType i = 0; i < size; ++i)
    {
        for (IdType j = 0; j < size; ++j)
        {
            IdType row = i * size + j;

            // Center element
            laplacian.values[idx] = 4.0;
            laplacian.col_idx[idx] = row;
            ++idx;

            // Left element
            if (j > 0)
            {
                laplacian.values[idx] = -1.0;
                laplacian.col_idx[idx] = row - 1;
                ++idx;
            }

            // Right element
            if (j < size - 1)
            {
                laplacian.values[idx] = -1.0;
                laplacian.col_idx[idx] = row + 1;
                ++idx;
            }

            // Top element
            if (i > 0)
            {
                laplacian.values[idx] = -1.0;
                laplacian.col_idx[idx] = row - size;
                ++idx;
            }

            // Bottom element
            if (i < size - 1)
            {
                laplacian.values[idx] = -1.0;
                laplacian.col_idx[idx] = row + size;
                ++idx;
            }

            laplacian.row_ptr[row + 1] = idx;
        }
    }

    return laplacian;
}

#endif //UTILS_H
