#include "mpi.h"
#include <stdlib.h>
#include <stdio.h>
#include <cstring>
#include <ctime>

/* Process mapping function */
int proc_map(int i, int size)
{
    return i%size;
}

int main(int argc, char** argv)
{
    int size, rank;
    MPI_Status Stat;

    int AROW = atoi(argv[1]);
    int ACOL = atoi(argv[2]);
    int BCOL = atoi(argv[3]);
    int MAX_VALUE = atoi(argv[4]);
 
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
 
    if (rank == 0)
    {
        int a[AROW][ACOL];
    	int b[ACOL][BCOL];
        int c[AROW][BCOL];
 
    	double timeSec;
    	clock_t begin, end;

        /* Generating Random Values for A & B Array*/
        srand(10);

        for (int i=0;i<AROW;i++)
        {
            for (int j=0;j<ACOL;j++)
            {
                a[i][j] = rand() % MAX_VALUE;
            }
        }
 
        for (int i=0;i<ACOL;i++)
        {
            for (int j=0;j<BCOL;j++)
            {
                b[i][j] = rand() % MAX_VALUE;
            }
        }
 
	begin = clock(); 

        /* (1) Sending B Values to other processes */
        if (size > 1)
            MPI_Bcast(&b[0][0], BCOL*ACOL, MPI_INTEGER, 0, MPI_COMM_WORLD); 
 
        /* (2) Sending Required A Values to specific process */
        for (int i=0;i<AROW;i++)
        {
            int processor = proc_map(i, size);
            if (processor != 0) 
                MPI_Send(a[i], ACOL, MPI_INTEGER, processor, (100*(i+1)), MPI_COMM_WORLD);
        }

        for (int i=0;i<AROW;i+=size) 
        {
            for (int j=0;j<BCOL;j++)
            {
                int sum = 0;
                for (int z=0;z<ACOL;z++)
                {
                    sum = sum + (a[i][z] * b[z][j] );
                }
                c[i][j] = sum;
            }
        }
 
        /* (3) Gathering the result from other processes*/
        for (int i=0;i<AROW;i++)
        {
            int source_process = proc_map(i, size);
            if (source_process != 0)
                MPI_Recv(c[i], BCOL, MPI_INTEGER, source_process, i, MPI_COMM_WORLD, &Stat);
        }
		
	end = clock();

    	/* Printing the Result */
		
		/*
        printf("Matrix A :\n");
        for (int i=0;i<AROW;i++)
        {
            for (int j=0;j<ACOL;j++)
            {
                printf("%3d ", a[i][j]);
            }
            printf("\n");
        }
 
        printf("\nMatrix B :\n");
        for (int i=0;i<ACOL;i++)
        {
            for (int j=0;j<BCOL;j++)
            {
                printf("%3d ", b[i][j]);
            }
            printf("\n");
        }
        printf("\n");

		printf("\nMatrix C :\n");
        for (int i=0;i<AROW;i++)
        {
            for (int x=0;x<BCOL;x++)
            {
                printf("%3d ", c[i][x]);
            }
            printf("\n");
        }
		*/
	timeSec = (end - begin)/static_cast<double>(CLOCKS_PER_SEC);
	printf("Elapsed time with %d processors is %f seconds\n", size, timeSec);
    }
    else
    {
	int b[ACOL][BCOL];
        /* (1) Each process get B Values from Master */
        MPI_Bcast(&b[0][0], BCOL*ACOL, MPI_INTEGER, 0, MPI_COMM_WORLD); 
 
        /* (2) Get Required A Values from Master then Compute the result */
        for (int i=0;i<AROW;i++)
        {
            int processor = proc_map(i, size);
            if (rank == processor)
            {
                int c[BCOL];
                int buffer[ACOL];
                MPI_Recv(buffer, ACOL, MPI_INTEGER, 0, (100*(i+1)), MPI_COMM_WORLD, &Stat);
                for (int j=0;j<BCOL;j++)
                {
                    int sum = 0;
                    for (int z=0;z<ACOL;z++)
                    {
                        sum = sum + (buffer[z] * b[z][j] );
                    }
                    c[j] = sum;
                }
                MPI_Send(c, BCOL, MPI_INTEGER, 0, i, MPI_COMM_WORLD);
            }
        }
    }
 
    MPI_Finalize();
    return 0;
}
