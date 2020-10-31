/*
* ECSE420 LAB2: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void synthesis_512(float* u_cuda, float* u1_cuda, float* u2_cuda, float p, float eta, float g, int n, int num_of_elems_per_thread, int num_of_threads)
{
	//update all the interior elements
	int index_temp = threadIdx.x + blockIdx.x * blockDim.x;
	index_temp = index_temp * num_of_elems_per_thread;

	for (int m = 0; m < num_of_elems_per_thread; m++) {
		int index = index_temp + m;
		if (index < n * n) {
			int i = index / n; //row
			int j = index % n; //column
			if (i < (n - 1) && i > 0 && j < (n - 1) && j > 0) {
				u_cuda[index] = (p * (u1_cuda[(i - 1) * n + j] + u1_cuda[(i + 1) * n + j] + u1_cuda[i * n + (j - 1)] + u1_cuda[i * n + (j + 1)] - 4 * u1_cuda[i * n + j]) + 2 * u1_cuda[i * n + j] - (1 - eta) * u2_cuda[i * n + j])
					/ (1 + eta);
			}

			//update boundaries
			if (j < (n - 1) && j > 0) {
				if (i == 0) {
					u_cuda[index] = g * u_cuda[1 * n + j];
				}
				else if (i == (n - 1)) {
					u_cuda[index] = g * u_cuda[(n - 2) * n + j];
				}
			}
			if (i < (n - 1) && i > 0) {
				if (j == 0) {
					u_cuda[index] = g * u_cuda[i * n + 1];
				}
				else if (j == (n - 1)) {
					u_cuda[index] = g * u_cuda[i * n + (n - 2)];
				}
			}

			//update corners
			if (i == 0 && j == 0) {
				u_cuda[index] = g * u_cuda[1 * n + 0];
			}
			else if (i == (n - 1) && j == 0) {
				u_cuda[index] = g * u_cuda[(n - 2) * n + 0];
			}
			else if (i == 0 && j == (n - 1)) {
				u_cuda[index] = g * u_cuda[0 * n + (n - 2)];
			}
			else if (i == (n - 1) && j == (n - 1)) {
				u_cuda[index] = g * u_cuda[(n - 1) * i + (n - 2)];
			}

		}
	}
}

int process_synthesis_512(int argc, char* argv[]) {
	//assign each node to a thread

	if (argc != 2)
		return 0;

	// get arguments from command line
	int num_of_iterations = atoi(argv[1]);
	int mode = 1; //change as needded. can be 1, 2, 3, 4 or 5
	int n = 512; //array is n x n

	float* u2; //previous previous array
	float* u1; // previous array
	float* u; //array we will work on
	unsigned int arraySize = (n * n) * sizeof(float);

	u2 = (float*)malloc(arraySize);
	u1 = (float*)malloc(arraySize);
	u = (float*)malloc(arraySize);

	for (int a = 0; a < (n * n); a++) {
		u2[a] = 0;
		u1[a] = 0;
		u[a] = 0;
	}

	u1[(n / 2) * n + n / 2] = 1; //simulated hit on the drum at (n/2, n/2)

	float p = 0.5;
	float eta = 0.0002;
	float g = 0.75; //boundary gain
	
	//defining device vars
	float* u2_cuda, *u1_cuda, *u_cuda;

	cudaMalloc((void**)&u2_cuda, arraySize);
	cudaMalloc((void**)&u1_cuda, arraySize);
	cudaMalloc((void**)&u_cuda, arraySize);

	//number of threads
	int num_of_threads = 0;
	int num_of_blocks = 0;
	int num_of_elems_per_thread = 0;

	if (mode == 1) { //16 blocks, 16 elements by thread, 32 rows per block, 32 threads per row
		num_of_threads = 1024;
		num_of_blocks = 16;
		num_of_elems_per_thread = 16;
	}
	else if (mode == 2) { //512 blocks, 1 element by thread, 1 row per block, 512 threads per row
		num_of_threads = 512;
		num_of_blocks = 512;
		num_of_elems_per_thread = 1;
	}
	else if (mode == 3) { //64 blocks, 4 elements by thread, 8 rows per block, 128 threads per row
		num_of_threads = 1024;
		num_of_blocks = 64;
		num_of_elems_per_thread = 4;
	}
	else if (mode == 4) { //64 blocks, 4 elements by thread, 8 rows per block, 128 threads per row
		num_of_threads = 512;
		num_of_blocks = 4;
		num_of_elems_per_thread = 128;
	}
	else if (mode == 5) { //64 blocks, 4 elements by thread, 8 rows per block, 128 threads per row
		num_of_threads = 1024;
		num_of_blocks = 256;
		num_of_elems_per_thread = 1;
	}

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

		synthesis_512 << < num_of_blocks, num_of_threads >> > (u_cuda, u1_cuda, u2_cuda, p, eta, g, n, num_of_elems_per_thread, num_of_threads);
		cudaDeviceSynchronize();

		//get new u
		cudaMemcpy(u, u_cuda, arraySize, cudaMemcpyDeviceToHost);

		//print out position u[n/2][n/2]
		printf("iteration %d: position (%d, %d) = %f \n", t, n / 2, n / 2, u[(n / 2) * n + (n / 2)]);

		//update u1 and u2
		memcpy(u2, u1, arraySize);
		memcpy(u1, u, arraySize);

	}

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Grid 512x512: thread count is %d, ran in %f milliseconds\n", num_of_threads, memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	//free cuda memory
	cudaFree(u_cuda);
	cudaFree(u1_cuda);
	cudaFree(u2_cuda);

	//free memory
	free(u2);
	free(u1);
	free(u);

	return 0;
}

int main(int argc, char* argv[]) { return process_synthesis_512(argc, argv); }
