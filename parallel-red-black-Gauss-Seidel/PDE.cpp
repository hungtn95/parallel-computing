#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <ctime>
#include <string.h>
#include <cmath>
#include <iostream> 
#include <string>

void clean_up(double** &matrix, int n) {
	for (int i = 0; i < n; i++) {
		delete[] matrix[i];
	}
	delete[] matrix;
}

void initialize_matrix(double** &matrix, int n, int m) {
	matrix = new double*[n];
	for (int i = 0; i < n; i++) {
		matrix[i] = new double[m];
	}
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < m; j++) {
			matrix[i][j] = 0;
		}
	}
}

void initialize_vector(double* &vector, int n) {
	vector = new double[n];
	for (int i = 0; i < n; i++) {
		vector[i] = 0;
	}
}

int index(int x, int y, int n) {
	return (n*y + x) / 2;
}

double f(double x, double y) {
	return (x*x + y*y)*exp(x*y);
}

int proc_map(int i, int size) {
	return (i-1) % size;
}

int count_red(int y, int n) {
	if (n % 2 != 0 && y % 2 != 0)
		return n / 2;
	return n / 2 - 1;
}

int count_black(int y, int n) {
	if (n % 2 != 0 && y % 2 == 0)
		return n / 2;
	return n / 2 - 1;
}

int start_red(int y) {
	if (y % 2 == 0)
		return 2;
	return 1;
}

int start_black(int y) {
	if (y % 2 == 0)
		return 1;
	return 2;
}

int neighbor_black(int y, int k) {
	return (y % 2 == 0) ? k - 1 : k + 1;
}

int neighbor_red(int y, int k) {
	return (y % 2 == 0) ? k + 1 : k - 1;
}

int main(int argc, char** argv)
{
	int n = atoi(argv[1]);
	double tol = atof(argv[2]);

	int size, rank;
	MPI_Status Stat;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double dx = 1.0 / n;
	double dy = 1.0 / n;
	double alpha = 2 * (pow(dx / dy, 2) + 1);
	double residual;

	if (rank == 0) {
		double timeSec;
		clock_t begin, end;
		double* red = NULL;
		double* black = NULL;
		double** mesh = NULL;
		int num_iter = 0;
		initialize_matrix(mesh, n, n);
		initialize_vector(red, n*n / 2);
		initialize_vector(black, n*n / 2);
		begin = clock(); 
		do {
			residual = 0;
			// Send Neighbor Black Nodes
			for (int y = 1; y <= n - 2; y++) {
				int dst = proc_map(y, size);
				if (dst != 0) {
					int num_red = count_red(y, n);
					int i = 0;
					double buffer[num_red][7];
					for (int x = start_red(y); x <= n - 2; x += 2) {
						int k = index(x, y, n);
						buffer[i][0] = black[k];
						buffer[i][1] = black[neighbor_black(y, k)];
						buffer[i][2] = black[k - n / 2];
						buffer[i][3] = black[k + n / 2];
						buffer[i][4] = 1.0*x;
						buffer[i][5] = 1.0*y;
						buffer[i][6] = red[k];
						i++;
					}
					MPI_Send(&buffer[0][0], num_red*7, MPI_DOUBLE, dst, (100*(y+1)), MPI_COMM_WORLD);
					//printf("%d sent %d %d red nodes\n", 0, dst, num_red);
				}
			}					

			// Update Red Nodes in Processor 0
			for (int y = 1; y <= n - 2; y++) {
				int src = proc_map(y, size);
				if (src == 0) {
					for (int x = start_red(y); x <= n - 2; x += 2) {
						int k = index(x, y, n);
						double previous = red[k];
						red[k] = (black[k] + black[neighbor_black(y, k)] + pow(dx / dy, 2)*(black[k - n / 2] + black[k + n / 2]) + pow(dx,2)*f(x*dx, y*dy)) / alpha;
						residual += pow(red[k] - previous, 2);
					}
				}
			}

			// Receive and Update Red Nodes from other processors
			for (int y = 1; y <= n - 2; y++) {
				int src = proc_map(y, size);
				if (src != 0) {
					int num_red = count_red(y, n);
					double buffer[num_red+1];
					MPI_Recv(buffer, num_red+1, MPI_DOUBLE, src, y, MPI_COMM_WORLD, &Stat);
					//printf("%d received %d red nodes\n", rank, num_red);
					int i = 0;
					for (int x = start_red(y); x <= n - 2; x += 2) {
						int k = index(x, y, n);
						red[k] = buffer[i];
						i++;
					}
					residual += buffer[num_red];
				}
			}

			// Send Neighbor Red Nodes
			for (int y = 1; y <= n - 2; y++) {
				int dst = proc_map(y, size);
				if (dst != 0) {
					int num_black = count_black(y, n);
					int i = 0;
					double buffer[num_black][7];
					for (int x = start_black(y); x <= n - 2; x += 2) {
						int k = index(x, y, n);
						buffer[i][0] = red[k];
						buffer[i][1] = red[neighbor_red(y, k)];
						buffer[i][2] = red[k - n / 2];
						buffer[i][3] = red[k + n / 2];
						buffer[i][4] = 1.0*x;
						buffer[i][5] = 1.0*y;
						buffer[i][6] = black[k];
						i++;
					}
					MPI_Send(&buffer[0][0], num_black*7, MPI_DOUBLE, dst, 1000*y, MPI_COMM_WORLD);
					//printf("%d sent %d %d black nodes\n", 0, dst, num_black);
				}
			}
			// Update Black Nodes in Processor 0
			for (int y = 1; y <= n - 2; y++) {
				int src = proc_map(y, size);
				if (src == 0) {
					for (int x = start_black(y); x <= n - 2; x += 2) {
						int k = index(x, y, n);
						double previous = black[k];
						black[k] = (red[k] + red[neighbor_red(y, k)] + pow(dx / dy, 2)*(red[k - n / 2] + red[k + n / 2]) + pow(dx, 2)*f(x*dx, y*dy)) / alpha;
						residual += pow(black[k] - previous, 2);
					}
				}
			}

			// Receive and Update Black Nodes from other processors
			for (int y = 1; y <= n - 2; y++) {
				int src = proc_map(y, size);
				if (src != 0) {
					int num_black = count_black(y, n);
					double buffer[num_black+1];
					MPI_Recv(buffer, num_black+1, MPI_DOUBLE, src, 1000+y, MPI_COMM_WORLD, &Stat);
					//printf("%d received %d black nodes\n", rank, num_black);
					int i = 0;
					for (int x = start_black(y); x <= n - 2; x += 2) {
						int k = index(x, y, n);
						black[k] = buffer[i];
						i++;
					}
					residual += buffer[num_black];
				}
			}
			MPI_Bcast(&residual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
			/*
			printf("%d,%4f,\n", num_iter, sqrt(residual));
			num_iter++;
			*/
		} while (sqrt(residual) > tol);

		end = clock(); 

		timeSec = (end - begin)/static_cast<double>(CLOCKS_PER_SEC);
		printf("Elapsed time is %f seconds\n", timeSec);

		/*
		for (int y = 1; y <= n - 2; y++) {
			for (int x = start_red(y); x <= n - 2; x += 2) {
				mesh[y][x] = red[index(x, y, n)];
			}
			for (int x = start_black(y); x <= n - 2; x += 2) {
				mesh[y][x] = black[index(x, y, n)];
			}
		}
		for (int y = 0; y < n; y++) {
			for (int x = 0; x < n; x++) {
				printf("%.4f ", mesh[y][x]);
			}
			printf("\n");
		}

		clean_up(mesh, n);
		*/
		delete[] red;
		delete[] black;
	} 
	else {
		// Other processor
		do {
			residual = 0;
			// Update Red Nodes in other processors
			for (int y = 1; y <= n - 2; y++) {
				int processor = proc_map(y, size);
				if (processor == rank) {
					int num_red = count_red(y, n);
					double in[num_red][7];
					double out[num_red+1];
					out[num_red] = 0;
					MPI_Recv(&in[0][0], num_red*7, MPI_DOUBLE, 0, (100*(y+1)), MPI_COMM_WORLD, &Stat);
					for (int i = 0; i < num_red; i++) {
						out[i] = (in[i][0] + in[i][1] + pow(dx / dy, 2)*(in[i][2] + in[i][3]) + pow(dx, 2)*f(in[i][4]*dx, in[i][5]*dy)) / alpha;
						out[num_red] += pow(out[i] - in[i][6], 2); 
					}
					MPI_Send(out, num_red+1, MPI_DOUBLE, 0, y, MPI_COMM_WORLD);
				}
			}

			// Update Black Nodes in other processors
			for (int y = 1; y <= n - 2; y++) {
				int processor = proc_map(y, size);
				if (processor == rank) {
					int num_black = count_black(y, n);
					double in[num_black][7];
					double out[num_black+1];
					out[num_black] = 0;
					MPI_Recv(&in[0][0], num_black*7, MPI_DOUBLE, 0, 1000*y, MPI_COMM_WORLD, &Stat);
					//printf("%d received %d black nodes\n", rank, num_black);
					for (int i = 0; i < num_black; i++) {
						out[i] = (in[i][0] + in[i][1] + pow(dx / dy, 2)*(in[i][2] + in[i][3]) + pow(dx, 2)*f(in[i][4]*dx, in[i][5]*dy)) / alpha;
						out[num_black] += pow(out[i] - in[i][6], 2); 
					}
					MPI_Send(out, num_black+1, MPI_DOUBLE, 0, 1000+y, MPI_COMM_WORLD);
				}
			}

			MPI_Bcast(&residual, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		} while (sqrt(residual) > tol);
	}

	MPI_Finalize();
	return 0;
}

