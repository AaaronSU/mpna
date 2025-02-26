#include <iostream>
#include <vector>
#include "CSRMatrix.h"
#include "ConjugateGradient.h"
#include "GMRES.h"

int main()
{
    // Créer une matrice de Laplacien 2D (taille x taille)
    int size = 10;
    auto A = createLaplacian2D<int, double>(size, 0.0);

    // Construire le vecteur du second membre b, rempli de 1.0
    std::vector<double> b(A.n, 1.0);

    // Définir la solution initiale x0 = 0
    std::vector<double> x(A.n, 0.0);

    double tol = 1e-8;
    int max_iter = 1000;
    int restart = 20;     // Nombre d'itérations entre redémarrages (pour gmres)

    int iter_used = conjugateGradient(A, b, x, tol, max_iter);

    std::cout << "Le gradient conjugué a terminé en " 
              << iter_used << " itérations." << std::endl;

    std::cout << "x[0.." << A.n-1 << "] = ";
    for (int i = 0; i < A.n; ++i)
    {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    std::fill(x.begin(), x.end(), 0.0);

    int iter_used_gmres = gmres(A, b, x, restart, tol, max_iter);

    std::cout << "GMRES a terminé en " 
              << iter_used_gmres << " itérations." << std::endl;

    // Afficher les résultats pour vérifier la solution
    std::cout << "x[0.." << A.n - 1 << "] = ";
    for (int i = 0; i < A.n; ++i)
    {
        std::cout << x[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
