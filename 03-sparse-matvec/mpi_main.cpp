#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unordered_map>
#include "CSRMatrix.h"
#include "COOVector.h"
#include "Jacobi.h"
#include "GaussSeidel.h"
extern "C" {
#include "mmio.h"
}

int main(int argc, char *argv[])
{
    // Initialisation de l'environnement MPI
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Déclaration de la matrice CSR et des dimensions
    CSRMatrix<int, double> A(0, 0);
    int M, N, nz;
    std::vector<int> full_row_ptr;
    std::vector<int> full_col_idx;
    std::vector<double> full_values;
    
    if (rank == 0) {
        // Si un fichier est fourni en argument, le lire
        if (argc >= 2) {
            FILE *f = std::fopen(argv[1], "r");
            if (f == nullptr) {
                std::perror("Error opening file");
                return EXIT_FAILURE;
            }
            MM_typecode matcode;
            if (mm_read_banner(f, &matcode) != 0) {
                std::fprintf(stderr, "Could not process Matrix Market banner.\n");
                return EXIT_FAILURE;
            }
            if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
                std::fprintf(stderr, "Error reading matrix size.\n");
                return EXIT_FAILURE;
            }
            std::vector<int> I(nz), J(nz);
            std::vector<double> val(nz);
            for (int i = 0; i < nz; i++) {
                fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
                I[i]--;
                J[i]--;
            }
            std::fclose(f);
            A = convertCOOtoCSR<int, double>(M, N, nz, I, J, val);
        } else {
            // Sinon, créer une matrice Laplacien 2D par défaut
            int mesh_size = 9;
            A = createLaplacian2D<int, double>(mesh_size, 1.0);
            M = A.n;
            N = A.n;
        }
        // Sauvegarder les données CSR complètes
        full_row_ptr = A.row_ptr;
        full_col_idx = A.col_idx;
        full_values = A.values;
    }

    // Diffusion des dimensions de la matrice à tous les processus
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Diffusion du tableau complet row_ptr à tous les processus
    if (rank != 0)
        full_row_ptr.resize(M + 1);
    MPI_Bcast(full_row_ptr.data(), M + 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Diffusion du nombre total de non-zéros
    if (rank == 0)
        nz = full_row_ptr[M];
    MPI_Bcast(&nz, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Calcul de l'intervalle de lignes attribué à chaque processus
    int rows_per_proc = M / size;
    int remainder = M % size;
    int local_start, local_end;
    if (rank < remainder) {
        local_start = rank * (rows_per_proc + 1);
        local_end = local_start + rows_per_proc + 1;
    } else {
        local_start = rank * rows_per_proc + remainder;
        local_end = local_start + rows_per_proc;
    }
    int local_n = local_end - local_start;
    
    // Calcul des compteurs d'envoi et des décalages pour chaque processus (basé sur full_row_ptr)
    std::vector<int> sendcounts(size), displs(size);
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int r_start, r_end;
            if (r < remainder) {
                r_start = r * (rows_per_proc + 1);
                r_end = r_start + rows_per_proc + 1;
            } else {
                r_start = r * rows_per_proc + remainder;
                r_end = r_start + rows_per_proc;
            }
            sendcounts[r] = full_row_ptr[r_end] - full_row_ptr[r_start];
            displs[r] = full_row_ptr[r_start];
        }
    }
    // Calcul du nombre de non-zéros local pour ce processus
    int local_nnz = full_row_ptr[local_end] - full_row_ptr[local_start];

    // Distribution des tableaux col_idx et values de manière distribuée
    std::vector<int> local_col_idx(local_nnz);
    std::vector<double> local_values(local_nnz);
    MPI_Scatterv(rank == 0 ? full_col_idx.data() : nullptr, 
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_INT, local_col_idx.data(), local_nnz, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rank == 0 ? full_values.data() : nullptr, 
                 rank == 0 ? sendcounts.data() : nullptr,
                 rank == 0 ? displs.data() : nullptr,
                 MPI_DOUBLE, local_values.data(), local_nnz, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Construction du tableau local row_ptr (indexation relative)
    std::vector<int> local_row_ptr(local_n + 1);
    for (int i = 0; i <= local_n; i++) {
        local_row_ptr[i] = full_row_ptr[local_start + i] - full_row_ptr[local_start];
    }

    // Construction du vecteur dense x (vecteur de 1, présent sur tous les processus)
    std::vector<double> x(N, 1.0);
    // Multiplication matrice-vecteur locale : calcul du y_local
    std::vector<double> y_local(local_n, 0.0);
    for (int i = 0; i < local_n; i++) {
        double sum = 0.0;
        for (int j = local_row_ptr[i]; j < local_row_ptr[i+1]; j++) {
            int col = local_col_idx[j];
            sum += local_values[j] * x[col];
        }
        y_local[i] = sum;
    }
    
    // Rassemblement des résultats locaux y_local sur le processus 0
    std::vector<int> recvcounts(size), rdispls(size);
    if (rank == 0) {
        for (int r = 0; r < size; r++) {
            int r_start, r_end;
            if (r < remainder) {
                r_start = r * (rows_per_proc + 1);
                r_end = r_start + rows_per_proc + 1;
            } else {
                r_start = r * rows_per_proc + remainder;
                r_end = r_start + rows_per_proc;
            }
            recvcounts[r] = r_end - r_start;
            rdispls[r] = r_start;
        }
    }
    std::vector<double> y;
    if (rank == 0)
        y.resize(M, 0.0);
    MPI_Gatherv(y_local.data(), local_n, MPI_DOUBLE,
                rank == 0 ? y.data() : nullptr, 
                rank == 0 ? recvcounts.data() : nullptr, 
                rank == 0 ? rdispls.data() : nullptr, 
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Utilisation de y comme vecteur second membre b pour la résolution itérative (seulement sur le rang 0)
    int max_iter = 1000;
    double tolerance = 1e-6;
    int iter_jacobi = 0, iter_gs = 0;
    std::vector<double> x_jacobi, x_gs;
    if (rank == 0) {
        // Construction du vecteur b (ici, b = y, résultat de A*[1,1,...,1])
        std::vector<double> b = y;
        // La solution attendue est un vecteur rempli de 1
        std::vector<double> x_expected(M, 1.0);
        x_jacobi.resize(M, 0.0);
        x_gs.resize(M, 0.0);
        iter_jacobi = jacobi(M, A, b, x_jacobi, tolerance, max_iter);
        iter_gs = gaussSeidel(M, A, b, x_gs, tolerance, max_iter);
    }
    
    MPI_Finalize();
    return EXIT_SUCCESS;
}
