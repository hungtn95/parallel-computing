// Author:  Hung Nguyen
// Filename:  LDLLinearSolver.cpp
// Description:  LDL Linear Solver of an n-by-n matrix A.

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <iomanip>
#include <cmath>
#include <stdlib.h>
#include <time.h>
#include <ctime>
using namespace std;

// Function prototype.
void freeMem(double** A, int n, double* B, double* X);
void ldl(double** A, int n, double** L, double* D);
void fs(double* B, int n, double** L, double* Y);
void s(double* Y, int n, double* D, double* Z);
void bs(double* Z, int n, double** L, double* X);
void solve(double** A, int n, double* B, double* X);

// Main program.
int main(int argc, char* argv[])
{
    // Initialization.
    int i, j, n;
    double** A;
    double* B;
    double* X;

    // Test the functionality of the linear solver.
    if (argc == 5) {

        // Read in the matrix size.
        n = atoi(argv[1]);
        char* inafile = argv[2];
        char* inbfile = argv[3];
        char* outxfile = argv[4];

        // Create n-by-n matrix A.
        A = (double**)malloc(n*sizeof(double*));
        for (i = 0; i < n; ++i) {
            A[i] = (double*)malloc(n*sizeof(double));
        }

        // Initialize A to zeros matrix.
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                A[i][j] = 0.0;
            }
        }

        // Create vector B.
        B = (double*)malloc(n*sizeof(double));

        // Initialize B to zeros vector.
        for (i = 0; i < n; ++i) {
            B[i] = 0.0;
        }

        // Create vector X.
        X = (double*)malloc(n*sizeof(double));

        // Initialize X to zeros vector.
        for (i = 0; i < n; ++i) {
            X[i] = 0.0;
        }

        // Obtain names of output files.
        ifstream ina(inafile, ios::in);
        ifstream inb(inbfile, ios::in);
        ofstream outx(outxfile, ios::out);

        // Open output files for writing.
        if (!ina.is_open())
            cout << "Couldn't open output file for reading B." << endl;

        if (!inb.is_open())
            cout << "Couldn't open output file for reading A." << endl;

        if (!outx.is_open())
            cout << "Couldn't open output file for writing X." << endl;

        // Initialize A matrix to the matrix in the input file
        for (i = 0; i < n; ++i) {
            for (j = 0; j < n; ++j) {
                ina >> A[i][j];
            }
        }

        // Initialize B vector to the vector in the input file
        for (i = 0; i < n; ++i) {
            inb >> B[i];
        }

        // Solve for X.
        solve(A, n, B, X);

        // Write solution to file.
        outx << setprecision(16) << n << endl;
        for (i = 0; i < n; ++i) {
            outx << setprecision(16) << X[i] << " ";
        }
        outx << endl;

        // Close files.
        ina.close();
        inb.close();
        outx.close();

        // Free variables.
        freeMem(A, n, B, X);

    } else if (argc == 2) {

        char* outfile = argv[1];
        ofstream out(outfile, ios::out);
        if (!out.is_open())
            cout << "Couldn't open output file for writing result." << endl;

        for (n = 500; n <= 1000; n += 500) {
            // Create n-by-n matrix A.
            A = (double**)malloc(n*sizeof(double*));
            for (i = 0; i < n; ++i) {
                A[i] = (double*)malloc(n*sizeof(double));
            }

            // Initialize A to zeros matrix.
            for (i = 0; i < n; ++i) {
                for (j = 0; j < n; ++j) {
                    A[i][j] = 0.0;
                }
            }

            // Create vector B.
            B = (double*)malloc(n*sizeof(double));

            // Initialize B to zeros vector.
            for (i = 0; i < n; ++i) {
                B[i] = 0.0;
            }

            // Create vector X.
            X = (double*)malloc(n*sizeof(double));

            // Initialize X to zeros vector.
            for (i = 0; i < n; ++i) {
                X[i] = 0.0;
            }

            // Set random seed
            srand (time(NULL));

            // Initialize A to a random symmetric matrix.
            for (i = 0; i < n; ++i) {
                for (j = 0; j < n; ++j) {
                    if (i >= j) {
                        A[i][j] = rand() % 1000 + 1;
                        A[j][i] = A[i][j];
                    }
                }
            }

            // Initialize B to a random vector.
            for (i = 0; i < n; ++i) {
                B[i] = rand() % 1000 + 1;
            }

            // Solve for X.
            clock_t begin = clock();
            solve(A, n, B, X);
            clock_t end = clock();

            // Calculate timing result.
            double timeSec = (end - begin) / static_cast<double>(CLOCKS_PER_SEC);

            // Report timing result.
            out << "The LDL* solver takes " << timeSec << " seconds to solve for a " << n << "x" << n << " matrix." << endl;

            // Free variables.
            freeMem(A, n, B, X);

        }
        out.close();
    }
    else return(EXIT_FAILURE);

    // return
    return(EXIT_SUCCESS);
}

//////////////////////////////////////////////////////////////////////////////

void fs(double* B, int n, double** L, double* Y) {
    // Declare variables
    int i, j;
    double sum;

    Y[0] = B[0];
    // Perform forward substitution to find Y.
    for (i = 1; i < n; ++i) {
        sum = 0;
        for (j = 0; j < i; ++j) {
            sum = sum + L[i][j]*Y[j];
        }
        Y[i] = B[i] - sum;
    }
}

//////////////////////////////////////////////////////////////////////////////

void s(double* Y, int n, double* D, double* Z) {
    // Declare variables
    int i;

    // Perform substitution to find Z.
    for (i = 0; i < n; ++i) {
        Z[i] = Y[i] / D[i];
    }
}

//////////////////////////////////////////////////////////////////////////////

void bs(double* Z, int n, double** L, double* X) {
    // Declare variables
    int i, j;
    double sum;

    X[n-1] = Z[n-1];
    // Perform backward substitution to find X.
    for (i = n - 2; i >= 0; --i) {
        sum = 0.0;
        for (j = i + 1; j < n; ++j) {
            // Ltranspose[i][j] = L[j][i];
            sum = sum + L[j][i]*X[j];
        }
        X[i] = Z[i] - sum;
    }
}

//////////////////////////////////////////////////////////////////////////////

void ldl(double** A, int n, double** L, double* D){
    // Declare variables.
    int i, j, k;

    // Perform LDL factorization.
    for (i = 0; i < n; ++i) {
        // Compute multipliers.
        for (j = 0; j <= i; ++j) {
	    double d = 0;
	    double l = 0;
       	    // Apply transformation to the diagonal of matrix.
            for (k = 0; k < j; ++k) {
                d += L[j][k]*L[j][k]*D[k];
            	l += L[i][k]*D[k]*L[j][k];
            }
	    D[j] = A[j][j] - d;
	    L[i][j] = (A[i][j]-l)/D[j];
        }
    }
}

//////////////////////////////////////////////////////////////////////////////

void solve(double** A, int n, double* B, double* X) {
    int i, j;

    // Create L matrix.
    double** L = (double**)malloc(n*sizeof(double*));
    for (i = 0; i < n; ++i) {
        L[i] = (double*)malloc(n*sizeof(double));
    }

    // Create D vector.
    double* D = (double*)malloc(n*sizeof(double));

    // Create Y vector.
    double* Y = (double*)malloc(n*sizeof(double));

    // Create Z vector.
    double* Z = (double*)malloc(n*sizeof(double));

    // Compute LDL factorization of A.
    ldl(A, n, L, D);

    // Compute Y by forward substitution.
    fs(B, n, L, Y);

    // Compute Z by substitution.
    s(Y, n, D, Z);

    // Compute X by backward substitution.
    bs(Z, n, L, X);

    // Free L.
    for (i = 0; i < n; ++i) {
        free(L[i]);
    }
    free(L);

    // Free D.
    free(D);

    // Free Y.
    free(Y);

    // Free Z.
    free(Z);
}

//////////////////////////////////////////////////////////////////////////////

void freeMem(double**A, int n, double* B, double* X) {
    // Free A.
    for (int i = 0; i < n; ++i) {
        free(A[i]);
    }
    free(A);

    // Free B.
    free(B);

    // Free X.
    free(X);
}
