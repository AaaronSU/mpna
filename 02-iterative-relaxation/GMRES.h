#ifndef GMRES_H
#define GMRES_H

#include <vector>
#include <cmath>
#include "CSRMatrix.h"
#include "ConjugateGradient.h"

//----------------------------------------
// Implémentation de la méthode GMRES avec redémarrage
//----------------------------------------
template<typename IdType, typename ScalarType>
int gmres(
    const CSRMatrix<IdType, ScalarType>& A,
    const std::vector<ScalarType>& b,
    std::vector<ScalarType>& x,
    int restart,      // Nombre maximum d'itérations entre redémarrages
    ScalarType tol,   // Tolérance de convergence
    int max_iter      // Nombre maximum d'itérations totales
)
{
    int n = A.n;
    int iter_total = 0; // Compteur total d'itérations

    // Boucle extérieure pour les redémarrages
    while (iter_total < max_iter)
    {
        // Calcul du résidu r = b - A*x
        std::vector<ScalarType> Ax = matVec(A, x);
        std::vector<ScalarType> r(n, ScalarType(0));
        for (int i = 0; i < n; ++i)
        {
            r[i] = b[i] - Ax[i];
        }
        ScalarType beta = norm(r);
        if (beta < tol)
        {
            return iter_total; // Convergence atteinte
        }

        // Initialisation de la base de Krylov V (de dimension restart+1)
        std::vector<std::vector<ScalarType>> V(restart + 1, std::vector<ScalarType>(n, ScalarType(0)));
        for (int i = 0; i < n; ++i)
        {
            V[0][i] = r[i] / beta;
        }

        // Initialisation de la matrice de Hessenberg H (de dimension (restart+1) x restart)
        std::vector<std::vector<ScalarType>> H(restart + 1, std::vector<ScalarType>(restart, ScalarType(0)));

        // Vecteurs pour les rotations de Givens
        std::vector<ScalarType> cs(restart, ScalarType(0));
        std::vector<ScalarType> sn(restart, ScalarType(0));

        // Vecteur g pour le problème des moindres carrés
        std::vector<ScalarType> g(restart + 1, ScalarType(0));
        g[0] = beta;

        int j = 0;
        for (; j < restart && iter_total < max_iter; ++j)
        {
            // Calculer w = A * V[j]
            std::vector<ScalarType> w = matVec(A, V[j]);

            // Procédure de Gram-Schmidt modifiée
            for (int i = 0; i <= j; ++i)
            {
                H[i][j] = dot(w, V[i]);
                for (int k = 0; k < n; ++k)
                {
                    w[k] -= H[i][j] * V[i][k];
                }
            }
            H[j + 1][j] = norm(w);
            if (H[j + 1][j] != 0)
            {
                for (int i = 0; i < n; ++i)
                {
                    V[j + 1][i] = w[i] / H[j + 1][j];
                }
            }

            // Application des rotations de Givens antérieures à la colonne j de H
            for (int i = 0; i < j; ++i)
            {
                ScalarType temp = cs[i] * H[i][j] + sn[i] * H[i + 1][j];
                H[i + 1][j] = -sn[i] * H[i][j] + cs[i] * H[i + 1][j];
                H[i][j] = temp;
            }

            // Calcul de la nouvelle rotation de Givens pour éliminer H[j+1][j]
            ScalarType delta = std::sqrt(H[j][j] * H[j][j] + H[j + 1][j] * H[j + 1][j]);
            if (delta == 0)
            {
                cs[j] = 1;
                sn[j] = 0;
            }
            else
            {
                cs[j] = H[j][j] / delta;
                sn[j] = H[j + 1][j] / delta;
            }
            // Appliquer la rotation
            H[j][j] = cs[j] * H[j][j] + sn[j] * H[j + 1][j];
            H[j + 1][j] = 0;
            // Mettre à jour g
            ScalarType temp = cs[j] * g[j];
            g[j + 1] = -sn[j] * g[j];
            g[j] = temp;

            // Vérifier la convergence : norme résiduelle approximative = |g[j+1]|
            if (std::abs(g[j + 1]) < tol)
            {
                j++; // Augmente j pour refléter la dimension effective de l'espace de Krylov
                break;
            }
        }

        // Résolution du système triangulaire supérieur pour obtenir y
        std::vector<ScalarType> y(j, ScalarType(0));
        for (int i = j - 1; i >= 0; --i)
        {
            ScalarType sum = 0;
            for (int k = i + 1; k < j; ++k)
            {
                sum += H[i][k] * y[k];
            }
            y[i] = (g[i] - sum) / H[i][i];
        }

        // Mise à jour de la solution x = x + V[0:j] * y
        for (int i = 0; i < n; ++i)
        {
            for (int k = 0; k < j; ++k)
            {
                x[i] += y[k] * V[k][i];
            }
        }

        iter_total += j;
        // Vérifier si le nouveau résidu est en dessous de la tolérance
        Ax = matVec(A, x);
        r.assign(n, ScalarType(0));
        for (int i = 0; i < n; ++i)
        {
            r[i] = b[i] - Ax[i];
        }
        if (norm(r) < tol)
        {
            return iter_total;
        }
    }
    return iter_total;
}

#endif // GMRES_H
