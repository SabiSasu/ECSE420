/*
* ECSE420 LAB2: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__global__ void synthesis_512(float* u_cuda, float* u1_cuda, float* u2_cuda, float p, float eta, float g, int n, int num_of_threads, int mode)
{
	int index = threadIdx.x + blockIdx.x * blockDim.x;
	int i = index / n; //row
	int j = index % n; //column

	if (mode == 1) {
		printf("oops");
	}
	else if (mode == 2) {

		if (blockIdx.x != 0 && blockIdx.x != (n - 1)) { //not top or bottom row
			if (j != 0 && j != (n - 1)) {
				u_cuda[index] = (p * (u1_cuda[(i - 1) * n + j] + u1_cuda[(i + 1) * n + j] + u1_cuda[i * n + (j - 1)] + u1_cuda[i * n + (j + 1)] - 4 * u1_cuda[i * n + j]) + 2 * u1_cuda[i * n + j] - (1 - eta) * u2_cuda[i * n + j])
					/ (1 + eta);
			}
			else { //boundaries
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
			}
		}
		else {
			if (j != 0 && j != (n - 1)) {
				//boundaries
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
			}
			else {
				//corners
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
	else { //mode == 3
		printf("oops");
	}
}

int process_synthesis_512(int argc, char* argv[]) {
	//assign each node to a thread

	/*if (argc != 2)
		return 0;*/

	// get arguments from command line
	int num_of_iterations = 10; // atoi(argv[1]);

	int mode = 2; //change as needded. can be 1, 2, or 3

	float u2[512 * 512]; //previous previous array

	float u1[512 * 512]; //previous array

	float u[512 * 512]; //array we will work on

	int n = 512; //array is n x n

	for (int a = 0; a < (n * n); a++) {
		u2[a] = 0;
		u1[a] = 0;
		u[a] = 0;
	}

	u1[n / 2 * n + n / 2] = 1; //simulated hit on the drum at (n/2, n/2)

	float p = 0.5;
	float eta = 0.0002;
	float g = 0.75; //boundary gain

	//defining device vars
	float* u2_cuda;
	float* u1_cuda;
	float* u_cuda;

	cudaMalloc((void**)&u2_cuda, sizeof(u2));
	cudaMalloc((void**)&u1_cuda, sizeof(u1));
	cudaMalloc((void**)&u_cuda, sizeof(u));

	//number of threads
	int num_of_threads = 0;
	int num_of_blocks = 0;

	int num_of_threads_1 = 1024;
	int num_of_blocks_1 = 16;

	int num_of_threads_2 = 512;
	int num_of_blocks_2 = 512;

	int num_of_threads_3 = 1024;
	int num_of_blocks_3 = 64;

	if (mode == 1) { //16 blocks of 128 x 128 elements, 16 elements by thread
		num_of_threads = num_of_threads_1;
		num_of_blocks = num_of_blocks_1;
	}
	else if (mode == 2) { //512 blocks of 1 x 512 elements (1 row in each block), 1 element by thread
		num_of_threads = num_of_threads_2;
		num_of_blocks = num_of_blocks_2;
	}
	else { //64 blocks of 64 x 64 elements, 4 elements by thread
		num_of_threads = num_of_threads_3;
		num_of_blocks = num_of_blocks_3;
	}

	//start timer
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// synthesis
	for (int t = 0; t < num_of_iterations; t++) {
		//assign u, u1, u2 to cuda
		cudaMemcpy(u2_cuda, u2, sizeof(u2), cudaMemcpyHostToDevice);
		cudaMemcpy(u1_cuda, u1, sizeof(u1), cudaMemcpyHostToDevice);
		cudaMemcpy(u_cuda, u, sizeof(u), cudaMemcpyHostToDevice);

		synthesis_512 << < num_of_blocks, num_of_threads >> > (u_cuda, u1_cuda, u2_cuda, p, eta, g, n, num_of_threads, mode);
		cudaDeviceSynchronize();

		//get new u
		cudaMemcpy(u, u_cuda, sizeof(u), cudaMemcpyDeviceToHost);

		//print out position u[n/2][n/2]
		printf("iteration %d: position (%d, %d) = %f \n", t, n / 2, n / 2, u[(n / 2) * n + (n / 2)]);

		//update u1 and u2
		memcpy(u2, u1, sizeof(u1));
		memcpy(u1, u, sizeof(u1));
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

int main(int argc, char* argv[]) { return process_synthesis_512(argc, argv); }
