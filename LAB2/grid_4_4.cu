/*
* ECSE420 LAB2: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void synthesis(float* u_cuda, float* u1_cuda, float* u2_cuda, float p, float eta, float g, int n, int num_of_threads)
{
	//update all the interior elements
	int i = threadIdx.x / n; //row
	int j = threadIdx.x % n; //column
	if (i < (n - 1) && i > 0 && j < (n - 1) && j > 0) {
			u_cuda[threadIdx.x] = (p * (u1_cuda[(i - 1) * n + j] + u1_cuda[(i + 1) * n + j] + u1_cuda[i * n + (j - 1)] + u1_cuda[i* n + (j + 1)] - 4 * u1_cuda[i*n + j]) + 2 * u1_cuda[i*n + j] - (1 - eta) * u2_cuda[i*n + j])
				/ (1 + eta);
		}

	//update boundaries
	if (j < (n - 1) && j > 0) {
		if (i == 0) {
			u_cuda[threadIdx.x] = g * u_cuda[1 * n + j];
		}
		else if (i == (n - 1)) {
			u_cuda[threadIdx.x] = g * u_cuda[(n - 2) * n + j];
		}
	}
	if (i < (n - 1) && i > 0) {
		if (j == 0) {
			u_cuda[threadIdx.x] = g * u_cuda[i * n + 1];
		}
		else if (j == (n - 1)) {
			u_cuda[threadIdx.x] = g * u_cuda[i * n + (n - 2)];
		}
	}			

	//update corners
	if (i == 0 && j == 0) {
		u_cuda[threadIdx.x] = g * u_cuda[1 * n + 0];
	}
	else if (i == (n - 1) && j == 0) {
		u_cuda[threadIdx.x] = g * u_cuda[(n - 2) * n + 0];
	}
	else if (i == 0 && j == (n - 1)) {
		u_cuda[threadIdx.x] = g * u_cuda[0 * n + (n - 2)];
	}
	else if (i == (n - 1) && j == (n - 1)) {
		u_cuda[threadIdx.x] = g * u_cuda[(n - 1) * i + (n - 2)];
	}
}

int process_synthesis(int argc, char* argv[]) {
	//assign each node to a thread

	if (argc != 2)
		return 0;

	// get arguments from command line
	int num_of_iterations = atoi(argv[1]);

	float u2[16] = { //previous previous array
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};

	float u1[16] = { //previous array
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	int n = 4; //array is n x n

	u1[2*n + 2] = 1; //simulated hit on the drum at (n/2, n/2)

	float u[16] = { //array we will work on
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0,
		0, 0, 0, 0
	};
	float p = 0.5;
	float eta = 0.0002;
	float g = 0.75; //boundary gain

	//number of threads
	int num_of_threads = n * n;
	unsigned int arraySize = (n * n) * sizeof(float);

	//defining device vars
	float* u2_cuda;
	float* u1_cuda;
	float* u_cuda;

	cudaMalloc((void**)&u2_cuda, arraySize);
	cudaMalloc((void**)&u1_cuda, arraySize);
	cudaMalloc((void**)&u_cuda, arraySize);

	
	//start timer
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	
	// synthesis
	for (int t = 0; t < num_of_iterations; t++) {
		//assign u, u1, u2 to cuda
		cudaMemcpy(u2_cuda, u2, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(u1_cuda, u1, arraySize, cudaMemcpyHostToDevice);
		cudaMemcpy(u_cuda, u, arraySize, cudaMemcpyHostToDevice);

		synthesis << < 1, num_of_threads >> > (u_cuda, u1_cuda, u2_cuda, p, eta, g, n, num_of_threads);
		cudaDeviceSynchronize();

		//get new u
		cudaMemcpy(u, u_cuda, arraySize, cudaMemcpyDeviceToHost);

		//print out position u[n/2][n/2]
		printf("iteration %d: position (%d, %d) = %f \n", t, n / 2, n / 2, u[(n / 2) * n + (n / 2)]);

		//printing full matrix but only for debugging purposes
		/*printf("iteration %d: \n", t);
		printf("%f %f %f %f \n", u[0], u[1], u[2], u[3]);
		printf("%f %f %f %f \n", u[4], u[5], u[6], u[7]);
		printf("%f %f %f %f \n", u[8], u[9], u[10], u[11]);
		printf("%f %f %f %f \n", u[12], u[13], u[14], u[15]);*/

		//update u1 and u2
		memcpy(u2, u1, arraySize);
		memcpy(u1, u, arraySize);
	}
	

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Grid 4x4: thread count is %d, ran in %f milliseconds\n", num_of_threads, memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	//free cuda memory
	cudaFree(u_cuda);
	cudaFree(u1_cuda);
	cudaFree(u2_cuda);

	//free memory
	//free(u2);
	//free(u1);
	//free(u);

	return 0;
}
int main(int argc, char* argv[]) { return process_synthesis(argc, argv); }
