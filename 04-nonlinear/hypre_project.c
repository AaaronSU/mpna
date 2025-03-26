#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include "HYPRE_utilities.h"
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

double KAPPA0;
double SIGMA;
double BETA;
double Q_EXPONENT;
int N = 10;

#define FLAME_X   0.2
static double TOL     = 1.0e-10;
static int    MAXITER = 10000;

double source_term(int i)
{
    double x = (double)i / (double)N;
    if (x <= FLAME_X) return BETA;
    return 0.0;
}

double kappa_of_u(double u)
{
    if (u < 1.0e-14) u = 1.0e-14;
    return KAPPA0 * pow(u, Q_EXPONENT);
}

void BuildMatrixAndRHS_LinearImplicit(HYPRE_IJMatrix A, HYPRE_IJVector b,
                                        const double *u_current, double dt)
{
    int i;
    double dx  = 1.0 / (double)N;
    double dx2 = dx * dx;
    double zero = 0.0;
    for (i = 0; i <= N; i++)
        HYPRE_IJVectorSetValues(b, 1, &i, &zero);
    for (i = 0; i <= N; i++)
    {
        int ncols = 0;
        int cols[3];
        double vals[3];
        if (i == N)
        {
            ncols  = 1;
            cols[0] = N;
            vals[0] = 1.0;
            
            double rhs_val = 1.0;
            HYPRE_IJVectorSetValues(b, 1, &i, &rhs_val);
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
        }
        else if (i == 0)
        {
            double u0n = u_current[0];
            double kappa05 = 0.5 * (kappa_of_u(u_current[0]) + kappa_of_u(u_current[1]));
            double diag_val = 1.0/dt + 2.0 * kappa05/dx2 + SIGMA * pow(u0n, 3.0);
            double off_val  = -2.0 * kappa05/dx2;
            ncols = 2;
            cols[0] = 0;  vals[0] = diag_val;
            cols[1] = 1;  vals[1] = off_val;
            double rhs_val = (u0n / dt) + SIGMA * 1.0 + source_term(i);
            HYPRE_IJVectorSetValues(b, 1, &i, &rhs_val);
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
        }
        else if (i < N)
        {
            double ui_n   = u_current[i];
            double uim1_n = u_current[i-1];
            double uip1_n = u_current[i+1];
            double k_im05 = 0.5 * (kappa_of_u(ui_n) + kappa_of_u(uim1_n));
            double k_ip05 = 0.5 * (kappa_of_u(ui_n) + kappa_of_u(uip1_n));
            double diag_val  = 1.0/dt + (k_im05 + k_ip05)/dx2 + SIGMA * pow(ui_n, 3.0);
            double lower_val = -k_im05/dx2;
            double upper_val = -k_ip05/dx2;
            ncols = 3;
            cols[0] = i;    vals[0] = diag_val;
            cols[1] = i-1;  vals[1] = lower_val;
            cols[2] = i+1;  vals[2] = upper_val;
            double rhs_val = (ui_n/dt) + SIGMA * 1.0 + source_term(i);
            HYPRE_IJVectorSetValues(b, 1, &i, &rhs_val);
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
        }
    }
}

void BuildJacobianAndResidual_Newton(HYPRE_IJMatrix A, HYPRE_IJVector b,
                                       const double *u_current)
{
    int i;
    double dx  = 1.0 / (double)N;
    double dx2 = dx * dx;
    {
        double zero = 0.0;
        for (i = 0; i <= N; i++)
            HYPRE_IJVectorSetValues(b, 1, &i, &zero);
    }
    for (i = 0; i <= N; i++)
    {
        int ncols = 0;
        int cols[3];
        double vals[3];
        if (i == N)
        {
            ncols = 1;
            cols[0] = N; vals[0] = 1.0;
            double val = 1.0 - u_current[N];
            HYPRE_IJVectorSetValues(b, 1, &i, &val);
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
        }
        else if (i == 0)
        {
            double k05 = 0.5 * (kappa_of_u(u_current[0]) + kappa_of_u(u_current[1]));
            double diag_val = 2.0 * k05/dx2 + 4.0 * SIGMA * pow(u_current[0], 3.0);
            double off_val  = -2.0 * k05/dx2;
            ncols = 2;
            cols[0] = 0; vals[0] = diag_val;
            cols[1] = 1; vals[1] = off_val;
            double f0 = 2.0 * k05/dx2 * (u_current[0] - u_current[1])
                        + SIGMA * (pow(u_current[0], 4.0) - 1.0)
                        - source_term(0);
            double rhs_val = -f0;
            HYPRE_IJVectorSetValues(b, 1, &i, &rhs_val);
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
        }
        else if (i < N)
        {
            double ui   = u_current[i];
            double uim1 = u_current[i-1];
            double uip1 = u_current[i+1];
            double k_im05 = 0.5 * (kappa_of_u(ui) + kappa_of_u(uim1));
            double k_ip05 = 0.5 * (kappa_of_u(ui) + kappa_of_u(uip1));
            double diag_val = (k_im05 + k_ip05)/dx2 + 4.0 * SIGMA * pow(ui, 3.0);
            double low_val  = -k_im05/dx2;
            double up_val   = -k_ip05/dx2;
            ncols = 3;
            cols[0] = i;   vals[0] = diag_val;
            cols[1] = i-1; vals[1] = low_val;
            cols[2] = i+1; vals[2] = up_val;
            double Fi = - ( k_ip05 * (uip1 - ui) - k_im05 * (ui - uim1) )/dx2
                        + SIGMA * (pow(ui, 4.0) - 1.0)
                        - source_term(i);
            double rhs_val = -Fi;
            HYPRE_IJVectorSetValues(b, 1, &i, &rhs_val);
            HYPRE_IJMatrixSetValues(A, 1, &ncols, &i, cols, vals);
        }
    }
}

static void SolveWithBoomerAMG(HYPRE_IJMatrix A, HYPRE_IJVector b, HYPRE_IJVector x)
{
    HYPRE_ParCSRMatrix parA = NULL;
    HYPRE_ParVector    parb = NULL;
    HYPRE_ParVector    parx = NULL;
    HYPRE_IJMatrixGetObject(A, (void**)&parA);
    HYPRE_IJVectorGetObject(b, (void**)&parb);
    HYPRE_IJVectorGetObject(x, (void**)&parx);
    HYPRE_Solver solver;
    HYPRE_BoomerAMGCreate(&solver);
    HYPRE_BoomerAMGSetPrintLevel(solver, 0);
    HYPRE_BoomerAMGSetCoarsenType(solver, 6);
    HYPRE_BoomerAMGSetRelaxType(solver, 6);
    HYPRE_BoomerAMGSetNumSweeps(solver, 2);
    HYPRE_BoomerAMGSetMaxLevels(solver, 25);
    HYPRE_BoomerAMGSetTol(solver, 1e-12);
    HYPRE_BoomerAMGSetup(solver, parA, parb, parx);
    HYPRE_BoomerAMGSolve(solver, parA, parb, parx);
    HYPRE_BoomerAMGDestroy(solver);
}

void Solve_Nonlinear_LinearizedImplicit(double *u, double gamma)
{
    double umax  = 2.0;
    double dx    = 1.0 / (double)N;
    double dx2   = dx * dx;
    double dt    = gamma * gamma / (4.0 * SIGMA * pow(umax, 3.0) + 4.0 * kappa_of_u(umax)) * dx2;
    printf("Le pas du temps vaut %g\n", dt);
    double time_start = MPI_Wtime();
    int iter;
    for (iter = 0; iter < MAXITER; iter++)
    {
        HYPRE_IJMatrix A;
        HYPRE_IJVector b, x;
        int ilower = 0, iupper = N;
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);
        HYPRE_IJVectorInitialize(b);
        HYPRE_IJVectorInitialize(x);
        BuildMatrixAndRHS_LinearImplicit(A, b, u, dt);
        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJVectorAssemble(b);
        HYPRE_IJVectorAssemble(x);
        for (int i = 0; i <= N; i++)
        {
            HYPRE_IJVectorSetValues(x, 1, &i, &u[i]);
        }
        HYPRE_IJVectorAssemble(x);
        SolveWithBoomerAMG(A, b, x);
        double max_diff = 0.0;
        for (int i = 0; i <= N; i++)
        {
            double new_u;
            HYPRE_IJVectorGetValues(x, 1, &i, &new_u);
            double diff = fabs(new_u - u[i]);
            if (diff > max_diff) max_diff = diff;
            u[i] = new_u;
        }
        HYPRE_IJMatrixDestroy(A);
        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
        if (max_diff < TOL)
        {
            double time_end = MPI_Wtime();
            double elapsed = time_end - time_start;
            printf("La méthode converge à l'itération %d avec une erreur de %e et le temps écoulé vaut %g s\n", iter, max_diff, elapsed);
            return;
        }
    }
    double time_end = MPI_Wtime();
    double elapsed = time_end - time_start;
    printf("Le schéma implicite linéarisé a atteint %d itérations sans converger et le temps écoulé vaut %g s\n", MAXITER, elapsed);
}

void Solve_Nonlinear_Newton(double *u)
{
    double time_start = MPI_Wtime();
    for (int iter = 0; iter < MAXITER; iter++)
    {
        HYPRE_IJMatrix A;
        HYPRE_IJVector b, x;
        int ilower = 0, iupper = N;
        HYPRE_IJMatrixCreate(MPI_COMM_WORLD, ilower, iupper, ilower, iupper, &A);
        HYPRE_IJMatrixSetObjectType(A, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(A);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &b);
        HYPRE_IJVectorSetObjectType(b, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(b);
        HYPRE_IJVectorCreate(MPI_COMM_WORLD, ilower, iupper, &x);
        HYPRE_IJVectorSetObjectType(x, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(x);
        BuildJacobianAndResidual_Newton(A, b, u);
        HYPRE_IJMatrixAssemble(A);
        HYPRE_IJVectorAssemble(b);
        HYPRE_IJVectorAssemble(x);
        
        double zero = 0.0;
        for (int i = 0; i <= N; i++)
        {
            HYPRE_IJVectorSetValues(x, 1, &i, &zero);
        }
        HYPRE_IJVectorAssemble(x);
    
        SolveWithBoomerAMG(A, b, x);
        double max_d = 0.0;
        for (int i = 0; i <= N; i++)
        {
            double d_i;
            HYPRE_IJVectorGetValues(x, 1, &i, &d_i);
            if (fabs(d_i) > max_d) max_d = fabs(d_i);
            u[i] += d_i;
        }
        HYPRE_IJMatrixDestroy(A);
        HYPRE_IJVectorDestroy(b);
        HYPRE_IJVectorDestroy(x);
        if (max_d < TOL)
        {
            double time_end = MPI_Wtime();
            double elapsed = time_end - time_start;

            printf("La méthode converge à l'itération %d avec une erreur de %e et le temps écoulé vaut %g s\n", iter, max_d, elapsed);
            return;
        }
    }
    double time_end = MPI_Wtime();
    double elapsed = time_end - time_start;

    printf("Le schéma Newton a atteint %d itérations sans converger, avec le temps écoulé vaut %g s\n", MAXITER, elapsed);
}

void readConfig(const char *filename)
{
    FILE *file = fopen(filename, "r");
    if (!file) { perror("Erreur lors de l'ouverture du fichier de configuration"); exit(1); }
    char line[256];
    while (fgets(line, sizeof(line), file))
    {
        if (line[0] == '#' || strlen(line) < 3)
            continue;
        char key[64];
        double value;
        if (sscanf(line, "%63[^=]=%lf", key, &value) == 2)
        {
            if (strcmp(key, "KAPPA0") == 0)
                KAPPA0 = value;
            else if (strcmp(key, "SIGMA") == 0)
                SIGMA = value;
            else if (strcmp(key, "BETA") == 0)
                BETA = value;
            else if (strcmp(key, "Q_EXPONENT") == 0)
                Q_EXPONENT = value;
        }
    }
    fclose(file);
}

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    if (argc < 2)
    {
        printf("Usage: %s config.txt [gamma] [N]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }
    readConfig(argv[1]);
    double gamma = 0.1;
    if (argc >= 3) gamma = atof(argv[2]);
    if (argc >= 4) N = atoi(argv[3]);
    int step = (N > 3 ? N/3 : 1);


    printf("N=%d\n", N);

    double *u = malloc((N+1) * sizeof(double));
    if (u == NULL) {
        fprintf(stderr, "Memory allocation error\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i <= N; i++) {
        u[i] = 1.0;
    }
    printf("\n##### Schéma implicite linéarisé #####\n");

    Solve_Nonlinear_LinearizedImplicit(u, gamma);
    printf("Solution finale du schéma linéarisé :\n");
    for (int i = 0; i <= N; i += step)
    {
        double xx = (double)i / (double)N;
        printf("   i=%d, x=%.3f, u=%.6f\n", i, xx, u[i]);
    }

    for (int i = 0; i <= N; i++) {
        u[i] = 1.0;
    }
    printf("\n##### Schéma Newton-Raphson #####\n");
    Solve_Nonlinear_Newton(u);
    printf("Solution finale de Newton :\n");
    for (int i = 0; i <= N; i += step)
    {
        double xx = (double)i / (double)N;
        printf("   i=%d, x=%.3f, u=%.6f\n", i, xx, u[i]);
    }

    free(u);
    MPI_Finalize();
    return 0;
}
