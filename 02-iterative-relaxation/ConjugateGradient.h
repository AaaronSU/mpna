#ifndef CONJUGATEGRADIENT_H
#define CONJUGATEGRADIENT_H

#include <vector>
#include <cmath>
#include "CSRMatrix.h"

//----------------------------------------
// 1. Fonctions auxiliaires pour matrices creuses et vecteurs
//----------------------------------------

// Multiplication d’une matrice creuse A par un vecteur x : y = A * x
template <typename IdType, typename ScalarType>
std::vector<ScalarType> matVec(const CSRMatrix<IdType, ScalarType>& A, const std::vector<ScalarType>& x)
{
    std::vector<ScalarType> y(A.n, ScalarType(0));
    for (IdType i = 0; i < A.n; ++i)
    {
        ScalarType sum = 0;
        for (IdType j = A.row_ptr[i]; j < A.row_ptr[i + 1]; ++j)
        {
            sum += A.values[j] * x[A.col_idx[j]];
        }
        y[i] = sum;
    }
    return y;
}

// Produit scalaire dot(u, v) = ∑(u[i] * v[i])
template <typename ScalarType>
ScalarType dot(const std::vector<ScalarType>& u, const std::vector<ScalarType>& v)
{
    ScalarType result = 0;
    for (size_t i = 0; i < u.size(); ++i)
    {
        result += u[i] * v[i];
    }
    return result;
}

// Norme 2 du vecteur norm(u) = √(dot(u, u))
template <typename ScalarType>
ScalarType norm(const std::vector<ScalarType>& u)
{
    return std::sqrt(dot(u, u));
}

//----------------------------------------
// 2. Méthode du gradient conjugué
//----------------------------------------
template<typename IdType, typename ScalarType>
int conjugateGradient(
    const CSRMatrix<IdType, ScalarType>& A,
    const std::vector<ScalarType>& b,
    std::vector<ScalarType>& x,
    ScalarType tol,
    int max_iter)
{
    // r0 = b - A x0
    std::vector<ScalarType> Ax = matVec(A, x);
    std::vector<ScalarType> r(A.n);
    for (IdType i = 0; i < A.n; ++i)
    {
        r[i] = b[i] - Ax[i];
    }

    // p0 = r0
    std::vector<ScalarType> p = r;

    // On enregistre r_k^T r_k
    ScalarType rr = dot(r, r);

    // Boucle principale
    for (int k = 0; k < max_iter; ++k)
    {
        // A p_k
        std::vector<ScalarType> Ap = matVec(A, p);

        // alpha_k = (r_k^T r_k) / (p_k^T A p_k)
        ScalarType pAp = dot(p, Ap);
        ScalarType alpha = rr / pAp;

        // x_{k+1} = x_k + alpha_k * p_k
        for (IdType i = 0; i < A.n; ++i)
        {
            x[i] += alpha * p[i];
        }

        // r_{k+1} = r_k - alpha_k * A p_k
        for (IdType i = 0; i < A.n; ++i)
        {
            r[i] -= alpha * Ap[i];
        }

        // Vérification de la convergence
        ScalarType rr_new = dot(r, r);
        if (std::sqrt(rr_new) < tol)
        {
            // Convergence anticipée
            // Retourne le nombre d’itérations (après la (k+1)-ième)
            return k + 1;
        }

        // beta_k = (r_{k+1}^T r_{k+1}) / (r_k^T r_k)
        ScalarType beta = rr_new / rr;
        rr = rr_new;

        // p_{k+1} = r_{k+1} + beta_k * p_k
        for (IdType i = 0; i < A.n; ++i)
        {
            p[i] = r[i] + beta * p[i];
        }
    }

    // Si on atteint max_iter, on retourne max_iter
    return max_iter;
}

#endif // CONJUGATEGRADIENT_H
