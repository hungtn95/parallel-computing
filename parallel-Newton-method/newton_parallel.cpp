// Author:  Hung Nguyen
// Filename:  newton_parallel.cpp
// Description:  MPI code for Newton method.

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <ctime>
#include <string.h>
#include <cstring>
#include <mpi.h>

using namespace std;

// Function prototype.
void ldl(double** A, double** L, double* D, int n);
void forwardSubstitution(double* B, double** L, double* Y, int n);
void substitution(double* Y, double* D, double* Z, int n);
void backwardSubstitution(double* Z, double** L, double* X, int n);
void scalarMulti(double* A, double* X, int n, double scalar);
void vectorSubtract(double* A, double* B, double* X, int n);
void vectorAdd(double* A, double* B, double* X, int n);
double normFunction(double* A, int n);
void solve(double** A, double* B, double* X, int n);
double costFunction(double* d_x, int n);
void costFunctionGradient(double* d_x, double* d_g, int local_size, int global_size, int rank, int g_size);
void costFunctionHessian(double* d_x, double** d_h, int local_size, int global_size, int rank, int g_size);

double costFunction(double* d_x, int n) {
    double f = 0;
    for (int i = 0; i < n-1; i++) {
        f += 100*(d_x[i]*d_x[i]-d_x[i+1])*(d_x[i]*d_x[i]-d_x[i+1]) + (d_x[i]-1)*(d_x[i]-1);
    }
    return f;
}

void costFunctionGradient(double* d_x, double* d_g, int local_size, int global_size, int rank, int g_size) {
    int elem_per_proc = global_size / g_size;
    int start = elem_per_proc*rank;
    int end = start + local_size;
    int local_index = 0;

    for (int i = start; i < end; i++) {
        d_g[local_index] = (-200)*(d_x[i-1]*d_x[i-1]-d_x[i]) + 400*(d_x[i]*d_x[i]-d_x[i+1])*d_x[i] + 2*(d_x[i]-1);
        local_index++;
    }

    if (rank == 0)
        d_g[0] = 400*(d_x[0]*d_x[0]-d_x[1])*d_x[0] + 2*(d_x[0]-1);

    if (rank == g_size-1)
        d_g[local_size-1] = (-200)*(d_x[end-2]*d_x[end-2]-d_x[end-1]);
}


void costFunctionHessian(double* d_x, double** d_h, int local_size, int global_size, int rank, int g_size) {
    int elem_per_proc = global_size / g_size;
    int start = elem_per_proc*rank;
    int end = start + local_size;
    int local_index = 0;

    for (int i = start; i < end; i++) {
        d_h[0][local_index] = 200 + 1200*d_x[i]*d_x[i] - 400*d_x[i+1] + 2;
        d_h[1][local_index] = -400*d_x[i];
        local_index++;
    }

    if (rank == 0)
        d_h[0][0] = 1200*d_x[0]*d_x[0] - 400*d_x[1] + 2;

    if (rank == g_size-1) {
        d_h[0][local_size-1] = 200;
        d_h[1][local_size-1] = 0;
    }

    /*
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            d_h[i][j] = 0;
        }
    }

    d_h[0][0] = 1200*d_x[0]*d_x[0] - 400*d_x[1] + 2;
    d_h[0][1] = -400*d_x[0];
    d_h[1][0] = -400*d_x[0];
    d_h[n-1][n-1] = 200;

    for (int i = 1; i < n-1; i++) {
        d_h[i][i] = 200 + 1200*d_x[i]*d_x[i] - 400*d_x[i+1] + 2;
        d_h[i][i+1] = -400*d_x[i];
        d_h[i+1][i] = -400*d_x[i];
    }
    */
}

void scalarMulti(double* A, double* X, int n, double scalar) {
    for (int i = 0; i < n; i++) {
        X[i] = scalar*A[i];
    }
}

void vectorSubstract(double* A, double* B, double* X, int n) {
    for (int i = 0; i < n; i++) {
        X[i] = A[i] - B[i];
    }
}

void vectorAdd(double* A, double* B, double* X, int n) {
    for (int i = 0; i < n; i++) {
        X[i] = A[i] + B[i];
    }
}

double normFunction(double* A, int n) {
    double result = 0;
    for (int i = 0; i < n; i++) {
        result += A[i]*A[i];
    }
    return result;
}


// Main program.
int main(int argc, char** argv)
{
    int size, rank;
    MPI_Status Stat;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);



    // Initialization.
    int i, j, n, iter = 0;
    double tol = 0;
	double t1, t2;

    n = atoi(argv[1]);
    tol = atof(argv[2]);

    double* d_x = new double[n];;
    double* d_g = new double[n];
    double* delta_x = new double[n];
    double* negative_d_g = new double[n];
    double** d_h = new double*[2];

    int local_size;
    int elem_per_proc = n / size;
    int start = elem_per_proc*rank;

    if (rank == size-1)
        local_size = n - rank*elem_per_proc;
    else
        local_size = elem_per_proc;

    double* local_d_x = new double[local_size];
    double* local_d_g = new double[local_size];
    double** local_d_h = new double*[2];
    double* local_delta_x = new double[local_size];

    int counts[size];
    int displs[size];
    double local_sq_diff;
    double global_sq_diff;
    double local_sq_gradient;
    double global_sq_gradient;

    if (rank == 0) {
        // Create input vector.
        for (i = 0; i < n; ++i) {
            d_x[i] = 0;
        }
    }

    MPI_Bcast(d_x, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (i = 0; i < 2; ++i) {
        local_d_h[i] = new double[local_size];
    }

    for (i = 0; i < 2; ++i) {
        d_h[i] = new double[n];
    }

	if (rank == 0)
		t1 = MPI_Wtime(); 

    for (i = 0; i < size; i++) {
        displs[i] = i*elem_per_proc;
    }

    for (i = 0; i < size-1; i++) {
        counts[i] = elem_per_proc;
    }
    counts[size-1] = n - (size-1)*elem_per_proc;


    do {
        costFunctionGradient(d_x, local_d_g, local_size, n, rank, size);
        costFunctionHessian(d_x, local_d_h, local_size, n, rank, size);
        MPI_Gatherv(local_d_h[0], local_size, MPI_DOUBLE, d_h[0], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_d_h[1], local_size, MPI_DOUBLE, d_h[1], counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        MPI_Gatherv(local_d_g, local_size, MPI_DOUBLE, d_g, counts, displs, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            scalarMulti(d_g, negative_d_g, n, -1);
            double** temp = new double*[n];
            for (i = 0; i < n; i++) {
                temp[i] = new double[n];
            }
            for (i = 1; i < n; i++) {
                temp[i][i] = d_h[0][i];
                temp[i][i-1] = d_h[1][i-1];
            }
            temp[0][0] = d_h[0][0];
            solve(temp, negative_d_g, delta_x, n);

            for (i = 0; i < n; i++) {
                delete[] temp[i];
            }
            delete[] temp;
        }
        for (i = 0; i < local_size; i++) {
            local_d_x[i] = d_x[start + i];
        }

        MPI_Scatterv(delta_x, counts, displs, MPI_DOUBLE, local_delta_x, local_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        vectorAdd(local_d_x, local_delta_x, local_d_x, local_size);

        MPI_Allgatherv(local_d_x, local_size, MPI_DOUBLE, d_x, counts, displs, MPI_DOUBLE, MPI_COMM_WORLD);

        local_sq_diff = normFunction(local_delta_x, local_size);
        MPI_Allreduce(&local_sq_diff, &global_sq_diff, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        local_sq_gradient = normFunction(local_d_g, local_size);
        MPI_Allreduce(&local_sq_gradient, &global_sq_gradient, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        iter++;
		/*
		if (rank == 0) {
			printf("Iteration %d:\n", iter);
			printf("Norm of gradient = %f and norm of residual = %f\n", global_sq_gradient, global_sq_diff);
        	for (i = 0; i < n; ++i) {
           		printf("%f ", d_x[i]);
        	}
       	 	printf("\n\n");
		}
		*/
    } while (sqrt(global_sq_gradient) > tol);


	if (rank == 0)
		t2 = MPI_Wtime(); 

	printf( "Elapsed time for %d processors is %f\n", size, t2 - t1 );
    delete[] local_d_x;
    delete[] d_x;

    delete[] local_d_g;
    delete[] d_g;
    delete[] negative_d_g;

    delete[] local_delta_x;
    delete[] delta_x;

    for (i = 0; i < 2; ++i) {
        delete[] local_d_h[i];
    }
    delete[] local_d_h;

    for (i = 0; i < 2; ++i) {
        delete[] d_h[i];
    }
    delete[] d_h;


    // return
    MPI_Finalize();
    return(0);
}

//////////////////////////////////////////////////////////////////////////////

void forwardSubstitution(double* B, double** L, double* Y, int n) {
    // Declare variables
    int i, j;
    double sum;
    Y[0] = B[0];

    // Perform forward substitution to find Y.
    for (i = 1; i < n; ++i) {
        Y[i] = B[i] - L[i][i-1]*Y[i-1];
    }
}

//////////////////////////////////////////////////////////////////////////////

void substitution(double* Y, double* D, double* Z, int n) {
    // Declare variables
    int i;

    // Perform substitution to find Z.
    for (i = 0; i < n; ++i) {
        Z[i] = Y[i] / D[i];
    }
}

//////////////////////////////////////////////////////////////////////////////

void backwardSubstitution(double* Z, double** L, double* X, int n) {
    // Declare variables
    int i;
    X[n-1] = Z[n-1];

    // Perform backward substitution to find X.
    for (i = n - 2; i >= 0; --i) {
        X[i] = Z[i] - L[i+1][i]*X[i+1];
    }
}

//////////////////////////////////////////////////////////////////////////////

void ldl(double** A, double** L, double* D, int n){
    // Declare variables.
    int i;

    // Perform LDL factorization.
    for (i = 1; i < n; ++i) {
        D[i-1] = A[i-1][i-1] - L[i-1][i-2]*L[i-1][i-2]*D[i-2];
        L[i][i-1] = A[i][i-1]/D[i-1];
    }

    D[0] = A[0][0];
    D[n-1] = A[n-1][n-1] - L[n-1][n-2]*L[n-1][n-2]*D[n-2];
}

//////////////////////////////////////////////////////////////////////////////

void solve(double** A, double* B, double* X, int n) {
    int i;

    // Create L matrix.
    double** L = new double*[n];
    for (i = 0; i < n; ++i) {
        L[i] = new double[n];
    }

    // Create D vector.
    double* D = new double[n];

    // Create Y vector.
    double* Y = new double[n];

    // Create Z vector.
    double* Z = new double[n];

    // Compute LDL factorization of A.
    ldl(A, L, D, n);

    // Compute Y by forward substitution.
    forwardSubstitution(B, L, Y, n);

    // Compute Z by substitution.
    substitution(Y, D, Z, n);

    // Compute X by backward substitution.
    backwardSubstitution(Z, L, X, n);

    // Free L.
    for (i = 0; i < n; ++i) {
        delete[] L[i];
    }
    delete[] L;

    // Free D.
    delete[] D;

    // Free Y.
    delete[] Y;

    // Free Z.
    delete[] Z;
}
