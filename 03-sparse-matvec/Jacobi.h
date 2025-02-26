// Jacobi method for solving linear systems

#ifndef JACOBI_H
#define JACOBI_H

#include <vector>
#include <cmath>


#include "CSRMatrix.h"

template<typename IdType, typename ScalarType>
int jacobi(int n, const CSRMatrix<IdType, ScalarType>& A, const std::vector<ScalarType>& b, std::vector<ScalarType>& x, ScalarType tol, int max_iter)
{
    std::vector<ScalarType> x_old(n, 0.0);
    for (int iter = 0; iter < max_iter; ++iter)
    {
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
                    sum += A.values[j] * x_old[A.col_idx[j]];
                }
            }
            x[i] = (b[i] - sum) / diag;
        }

        // Simplest stopping criterion: check the norm of the difference
        ScalarType norm = 0.0;
        for (int i = 0; i < n; ++i)
        {
            norm += std::pow(x[i] - x_old[i], 2);
        }
        norm = std::sqrt(norm);

        if (norm < tol)
        {
            return iter;
        }

        x_old = x;
    }
    return max_iter;
}

#endif //JACOBI_H
