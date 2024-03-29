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

__global__ void logic_gate_unified(char* data, int file_length, char* outputData, int threadNum)
{

	for (int i = threadIdx.x + blockIdx.x; i < file_length; i += threadNum) {
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


int process_unified(int argc, char* argv[]) {

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

	char* data;
	char* output;
	cudaMallocManaged(&data, file_length * 6);
	cudaMallocManaged(&output, file_length * 2);
	fread(data, 1, file_length * 6, input_file);
	//printf("%s\n", data);
	//start timer
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// run
	logic_gate_unified << < num_blocks, num_threads_per_block >> > (data, file_length, output, num_threads_per_block);
	cudaDeviceSynchronize();

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Parallel Unified: file_length is %d, ran in %f milliseconds\n", file_length, memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	fwrite(output, 1, file_length * 2, output_file);

	fclose(input_file);
	fclose(output_file);
	cudaFree(data);
	cudaFree(output);
	return 0;
}

int main(int argc, char* argv[]) { return process_unified(argc, argv); }
