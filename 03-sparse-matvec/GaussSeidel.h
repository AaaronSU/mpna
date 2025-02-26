// Gauss-Seidel method for solving linear systems

#ifndef GAUSSSEIDEL_H
#define GAUSSSEIDEL_H

#include <vector>
#include <cmath>
#include "CSRMatrix.h"

template<typename IdType, typename ScalarType>
int gaussSeidel(int n, const CSRMatrix<IdType, ScalarType>& A, const std::vector<ScalarType>& b, std::vector<ScalarType>& x, ScalarType tol, int max_iter)
{
    for (int iter = 0; iter < max_iter; ++iter)
    {
        ScalarType norm = 0.0;
        for (int i = 0; i < n; ++i)
        {
            ScalarType sum = 0.0;
            ScalarType diag = 0.0;
            for (int j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j)
            {
                if (A.col_idx[j] == i)
                {
                    diag = A.values[j];
                }
                else
                {
                    sum += A.values[j] * x[A.col_idx[j]];
                }
            }
            ScalarType x_new = (b[i] - sum) / diag;
            norm += std::pow(x_new - x[i], 2);
            x[i] = x_new;
        }

        norm = std::sqrt(norm);
        if (norm < tol)
        {
            return iter;
        }
    }
    return max_iter;
}

#endif //GAUSSSEIDEL_H
