/*
* ECSE420 LAB0: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

__global__ void logic_gate(char* data, int file_length, char * outputData, int threadNum)
{

	for (int i = threadIdx.x + blockIdx.x; i < file_length; i += threadNum) {
		//printf("index: %d\n", i);
		int gateA = data[i*6] - '0';
		int gateB = data[i*6+2] - '0';
		int gateNum = data[i*6+4] - '0';
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
		outputData[i*2] = output +'0';
		outputData[i * 2 +1] = '\n';
	}
}


int process_unified(int argc, char* argv[]) {

	//if (argc != 4)
	//	return 0;

	// get arguments from command line
	char* input_filename = "test_data\\input_100000.txt";//argv[1];
	int file_length = 10000;//atoi(argv[2]);
	char* output_filename = "output_data\\output_100000.txt";//argv[3];

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

	int threadNum = file_length;
	int blockNum = 1;
	if (file_length > 1024) {
		threadNum = 1024;
		blockNum = file_length / 1024;
		printf("%d\n", threadNum);
		printf("%d\n", blockNum);
	}

	char* data;
	char* output;
	cudaMallocManaged(&data, file_length*6);
	cudaMallocManaged(&output, file_length*2);
	fread(data, 1, file_length*6, input_file);
	//printf("%s\n", data);
	//start timer
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// pool
	logic_gate << < blockNum, threadNum >> > (data, file_length, output, threadNum);
	cudaDeviceSynchronize();

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Pool: thread count is %d, ran in %f milliseconds\n", threadNum, memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	//free cuda memory
	//printf("output:\n %s\n", output);
	
	fwrite(output, 1, file_length*2, output_file);
	
	fclose(input_file);
	fclose(output_file);
	cudaFree(data);
	cudaFree(output);
	return 0;
}

//int main(int argc, char* argv[]) { return process_unified(argc, argv); }
