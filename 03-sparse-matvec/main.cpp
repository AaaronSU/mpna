#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
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
    CSRMatrix<int, double> A(0, 0);
    int M, N, nz;
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
        int mesh_size = 9;
        A = createLaplacian2D<int, double>(mesh_size, 1.0);
        M = A.n;
        N = A.n;
    }
    COOVector x_coo = createDenseCOOVector(N);
    COOVector y_coo = csrMatrixVectorMultiply(A, x_coo);
    std::vector<double> b(A.n, 0.0);
    for (size_t i = 0; i < y_coo.indices.size(); i++) {
        b[y_coo.indices[i]] = y_coo.values[i];
    }
    int max_iter = 1000;
    double tolerance = 1e-6;
    std::vector<double> x_jacobi(A.n, 0.0);
    int iter_jacobi = jacobi(A.n, A, b, x_jacobi, tolerance, max_iter);
    std::vector<double> x_gs(A.n, 0.0);
    int iter_gs = gaussSeidel(A.n, A, b, x_gs, tolerance, max_iter);

    std::cout << "\033[32mSuccès\033[0m" << std::endl;
    return EXIT_SUCCESS;
}