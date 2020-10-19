/*
* ECSE420 LAB1: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <Math.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

__global__ void logic_gate_explicit(char* data, int file_length, char* outputData, int threadNum)
{

	for (int i = threadIdx.x + (blockIdx.x * blockDim.x); i < file_length; i += threadNum) {
		//printf("index: %d\n", i);
		int gateA = data[i * 6] - '0';
		int gateB = data[i * 6 + 2] - '0';
		int gateNum = data[i * 6 + 4] - '0';
		//printf("%d, %d, %d\n", gateA, gateB, gateNum);
		int output = 0;
		switch (gateNum) {
		case AND: output = gateA & gateB;  break;
		case OR: output = gateA | gateB; break;
		case NAND: output = !(gateA & gateB); break;
		case NOR: output = !(gateA | gateB); break;
		case XOR: output = ((!gateA & gateB) | (gateA & !gateB)); break;
		case XNOR: output = !((!gateA & gateB) | (gateA & !gateB)); break;
		}
		//printf("%d\n", output);
		outputData[i * 2] = output + '0';
		outputData[i * 2 + 1] = '\n';
	}
}


int process_explicit(int argc, char* argv[]) {

	if (argc != 4)
		return 0;

	// get arguments from command line
	char* input_filename = argv[1];
	int file_length = atoi(argv[2]);
	char* output_filename = argv[3];

	FILE* input_file;
	FILE* output_file;

	if ((input_file = fopen(input_filename, "r")) == NULL) {
		printf("Error! opening file");
		// Program exits if file pointer returns NULL.
		exit(1);
	}

	if ((output_file = fopen(output_filename, "w")) == NULL) {
		printf("Error! opening file");
		// Program exits if file pointer returns NULL.
		exit(1);
	}

	int num_blocks = 1;
	int num_threads_per_block = file_length;

	if (file_length > 1024) {
		num_blocks = ((file_length - 1) / 1024) + 1; //1024 is the max number of threads in 1 block
		num_threads_per_block = ceil(file_length / num_blocks);
	}

	//create timer for data migration
	float memsettime2;
	cudaEvent_t start2, stop2;
	cudaEventCreate(&start2); cudaEventCreate(&stop2);
	cudaEventRecord(start2, 0);

	char* data;
	char* output;
	char* d_data;
	char* d_output;
	data = (char*)malloc(file_length * 6);
	output = (char*)malloc(file_length * 2);
	cudaMalloc(&d_data, file_length * 6);
	cudaMalloc(&d_output, file_length * 2);
	fread(data, 1, file_length * 6, input_file);
	cudaMemcpy(d_data, data, file_length * 6, cudaMemcpyHostToDevice);
	cudaMemcpy(d_output, output, file_length * 2, cudaMemcpyHostToDevice);

	cudaEventRecord(stop2, 0); cudaEventSynchronize(stop2);
	cudaEventElapsedTime(&memsettime2, start2, stop2);
	printf("Parallel Explicit data migration: file_length is %d, ran in %f milliseconds\n", file_length, memsettime2);
	cudaEventDestroy(start2); cudaEventDestroy(stop2);
	
	//start timer for execution runtime 
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// run
	logic_gate_explicit << < num_blocks, num_threads_per_block >> > (d_data, file_length, d_output, num_threads_per_block);

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Parallel Explicit: file_length is %d, ran in %f milliseconds\n", file_length, memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	//free cuda memory
	cudaMemcpy(output, d_output, file_length * 2, cudaMemcpyDeviceToHost);
	cudaFree(d_data);
	cudaFree(d_output);

	//write output to file
	fwrite(output, 1, file_length * 2, output_file);

	fclose(input_file);
	fclose(output_file);
	free(data);
	free(output);

	return 0;
}

int main(int argc, char* argv[]) { return process_explicit(argc, argv); }
