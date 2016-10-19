#include <stdio.h>
#include <float.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <cmath>

//#define VERBOSE
//#define DEBUG

/***********************************/
/* COST FUNCTION - CPU & GPU CASES */
/***********************************/
__host__ __device__ double Rosenbrock(double * h_x, int M) {

    // --- Rosenbrock function
    double sum = 0.f;
    for (int i=0; i<M-1; i++) {
        double temp1 = (h_x[i] * h_x[i] - h_x[i+1]);
        double temp2 = (h_x[i] - 1.f);
        sum = sum + 100.f * temp1 * temp1 + temp2 * temp2;
    }
    return sum;
}

/*******************************/
/* GRADIENT DESCENT - GPU CASE */
/*******************************/

__device__ double F_xn(double * d_x, int i, int dim) {
    if (i == 0)
        return -400.f * (d_x[1] - d_x[0] * d_x[0]) * d_x[0] + 2.f * (d_x[0] - 1.f);
    else if (i == dim-1) 
        return 200.f * (d_x[dim-1] - d_x[dim-2] * d_x[dim-2]);
    else
        return -400.f * d_x[i] * (d_x[i+1] - d_x[i] * d_x[i]) + 2.f * (d_x[i] - 1.f) + 200.f * (d_x[i] - d_x[i-1] * d_x[i-1]);
}

__device__ double F_xn_xn(double * d_x, int i, int dim) {
    if (i == 0)
        return 1200.f * d_x[0] * d_x[0] - 400.f * d_x[1] + 2;
    else if (i == dim-1)
        return 200;
    else 
        return 200.f + 1200.f * d_x[i] * d_x[i] - 400.f * d_x[i+1] + 2.f;
}

__device__ double F_xn_xn_plus_1(double * d_x, int i) {
    return -400.f * d_x[i];
}

__device__ double F_xn_xn_minus_1(double * d_x, int i) {
    return -400.f * d_x[i-1];
}

// --- Version using analytical gradient (Rosenbrock function)

__global__ void RosenbrockGradientGPU(double * d_x, double * d_g, double * d_g_norm, int dim, int offset) {
    int global_index = blockDim.x*blockIdx.x + threadIdx.x + offset;
    if (global_index < dim) {
        d_g[global_index] = -F_xn(d_x, global_index, dim);
		d_g_norm[global_index] = d_g[global_index]*d_g[global_index];
	}
}

__global__ void RosenbrockHessianGPU(double * d_x, double * d_h, int dim, int offset) {
    int global_index = blockDim.x*blockIdx.x + threadIdx.x + offset;
    if (global_index < dim) {
        d_h[global_index*dim + global_index] = F_xn_xn(d_x, global_index, dim);
        if (global_index < dim - 1)
            d_h[global_index*dim + global_index + 1] = F_xn_xn_plus_1(d_x, global_index);
        if (global_index > 0)
            d_h[global_index*dim + global_index - 1] = F_xn_xn_minus_1(d_x, global_index);
    }
}

__global__ void VectorNormGPU(double * v, double * v_norm, int dim, int offset) {
    int global_index = blockDim.x*blockIdx.x + threadIdx.x + offset;
    if (global_index < dim) 
        v_norm[global_index] = v[global_index]*v[global_index];
}

/*******************/
/* STEP - GPU CASE */
/*******************/

void ComputeGradientHessian(double * d_x, double * d_g, double * d_h, double * d_g_norm, int dim, int blocks, int threads) {
    double * d_x_cuda = NULL;
    double * d_g_cuda = NULL;
    double * d_g_norm_cuda = NULL;
    double * d_h_cuda = NULL;
    cudaSetDevice(0);
    cudaMalloc(&d_x_cuda, sizeof(double)*dim);
    cudaMalloc(&d_g_cuda, sizeof(double)*dim);
    cudaMalloc(&d_g_norm_cuda, sizeof(double)*dim);
    cudaMalloc(&d_h_cuda, sizeof(double)*dim*dim);
    cudaMemcpy(d_x_cuda, d_x, sizeof(double)*dim, cudaMemcpyHostToDevice);
    int offset = 0;
    while (offset < dim) {
        RosenbrockGradientGPU<<<blocks, threads>>>(d_x_cuda, d_g_cuda, d_g_norm_cuda, dim, offset);
        RosenbrockHessianGPU<<<blocks, threads>>>(d_x_cuda, d_h_cuda, dim, offset);
        cudaDeviceSynchronize();
        offset += blocks*threads;
    }

    cudaMemcpy(d_g, d_g_cuda, sizeof(double)*dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_g_norm, d_g_norm_cuda, sizeof(double)*dim, cudaMemcpyDeviceToHost);
    cudaMemcpy(d_h, d_h_cuda, sizeof(double)*dim*dim, cudaMemcpyDeviceToHost);

    cudaFree(d_x_cuda);
    cudaFree(d_g_cuda);
    cudaFree(d_g_norm_cuda);
    cudaFree(d_h_cuda);
}

void computeNorm(double * v, double * v_norm, int dim, int blocks, int threads) {
    double * v_norm_cuda = NULL;
    double * v_cuda = NULL;
    cudaSetDevice(0);
    cudaMalloc(&v_norm_cuda, sizeof(double)*dim);
    cudaMalloc(&v_cuda, sizeof(double)*dim);
    cudaMemcpy(v_cuda, v, sizeof(double)*dim, cudaMemcpyHostToDevice);
    int offset = 0;
    while (offset < dim) {
        VectorNormGPU<<<blocks, threads>>>(v_cuda, v_norm_cuda, dim, offset);
        cudaDeviceSynchronize();
        offset += blocks*threads;
    }
    cudaMemcpy(v_norm, v_norm_cuda, sizeof(double)*dim, cudaMemcpyDeviceToHost);
    cudaFree(v_norm_cuda);
    cudaFree(v_cuda);
}

double squareRootOfSum(double * v, int dim) {
	double sum = 0;
	for (int i = 0; i < dim; i++)
		sum += v[i];
	return sqrt(sum);
}

void vectorAdd(double* A, double* B, double* C, int dim) {
    for (int i = 0; i < dim; i++)
		C[i] = A[i] + B[i];
}

int transform(int i, int j, int n) {
    return i*n+j;
}

void forwardSubstitution(double* B, double* L, double* Y, int n) {
    // Declare variables
    int i;
    Y[0] = B[0];
    // Perform forward substitution to find Y.
    for (i = 1; i < n; ++i) {
        Y[i] = B[i] - L[transform(i, i-1, n)]*Y[i-1];
    }
}

void substitution(double* Y, double* D, double* Z, int n) {
    // Declare variables
    int i;
    // Perform substitution to find Z.
    for (i = 0; i < n; ++i) {
        Z[i] = Y[i] / D[i];
    }
}

void backwardSubstitution(double* Z, double* L, double* X, int n) {
    // Declare variables
    int i;
    X[n-1] = Z[n-1];
    // Perform backward substitution to find X.
    for (i = n - 2; i >= 0; --i) {
        X[i] = Z[i] - L[transform(i+1, i, n)]*X[i+1];
    }
}

void ldl(double* A, double* L, double* D, int n){
    // Declare variables.
    int i;
    // Perform LDL factorization.
    for (i = 1; i < n; ++i) {
        D[i-1] = A[transform(i-1, i-1, n)] - L[transform(i-1, i-2, n)]*L[transform(i-1, i-2, n)]*D[i-2];
        L[transform(i, i-1, n)] = A[transform(i, i-1, n)]/D[i-1];
    }
    D[0] = A[transform(0, 0, n)];
    D[n-1] = A[transform(n-1, n-1, n)] - L[transform(n-1, n-2, n)]*L[transform(n-1, n-2, n)]*D[n-2];
}

void solve(double* A, double* B, double* X, int n) {
    // Create L matrix.
    double* L = new double[n*n];
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
    delete[] L;
    // Free D.
    delete[] D;
    // Free Y.
    delete[] Y;
    // Free Z.
    delete[] Z;
}


/****************************************/
/* GRADIENT DESCENT FUNCTION - GPU CASE */
/****************************************/

// x0      - Starting point
// tol     - Termination tolerance
// maxiter - Maximum number of allowed iterations
// alpha   - Step size
// dxmin   - Minimum allowed perturbations

int main(int argc, char** argv)
{
	double timeSec;
	clock_t begin, end;
    int i;
	int iter = 0;
    int n = atoi(argv[1]);
	double tol = atoi(argv[2]);
    int blocks = atoi(argv[3]);
    int threads = atoi(argv[4]);

    double * d_x = new double[n];
    double * delta_x = new double[n];
    double * d_g = new double[n];
    double * d_g_norm = new double[n];
    double * delta_x_norm = new double[n];
    double * d_h = new double[n*n];
    
    for (i = 0; i < n; i++) {
        d_x[i] = i+1;
    }

	begin = clock();
	do {
    	ComputeGradientHessian(d_x, d_g, d_h, d_g_norm, n, blocks, threads);
		solve(d_h, d_g, delta_x, n);
		vectorAdd(d_x, delta_x, d_x, n);
		computeNorm(delta_x, delta_x_norm, n, blocks, threads);
		iter++;	
	} while (squareRootOfSum(d_g_norm, n) > tol);
	end = clock();

	timeSec = (end - begin)/static_cast<double>(CLOCKS_PER_SEC);
	printf("Blocks %d. Threads %d. Elapsed time is %f seconds\n", blocks, threads, timeSec);

    /*
	for (i = 0; i < n; i++) {
        printf("%4.2f ", d_x[i]);
    }
    printf("\n\n");
	*/
}