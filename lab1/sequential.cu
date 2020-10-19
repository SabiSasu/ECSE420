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


int process_sequential(int argc, char* argv[]) {

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
		exit(1);
	}

	if ((output_file = fopen(output_filename, "w")) == NULL) {
		printf("Error! opening file");
		exit(1);
	}
	
	char line[256];
	int i = 0;
	int nums[3] = { 0 };

	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	//start clock
	clock_t begin = clock();

	for (int i = 0; i < file_length; i++) {
		fgets(line, sizeof(line), input_file);
		int gateA = atoi(&line[0]);
		int gateB = atoi(&line[2]);
		int gateNum = atoi(&line[4]);

		int output = 0;
		switch (gateNum) {
		case AND: output = gateA & gateB;  break;
		case OR: output = gateA | gateB; break;
		case NAND: output = !(gateA & gateB); break;
		case NOR: output = !(gateA | gateB); break;
		case XOR: output = ((!gateA & gateB) | (gateA & !gateB)); break;
		case XNOR: output = !((!gateA & gateB) | (gateA & !gateB)); break;
		}
		fprintf(output_file, "%d\n", output);
	}
	//C timer library clock... doesnt work
	clock_t end = clock();
	long time_spent = ((double)end - begin) / CLOCKS_PER_SEC * 10000;
	printf("Execution time: %f\n",  time_spent);
	//cuda clock timer.... sorta works
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Cuda timer execution time %f milliseconds\n", memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	fclose(input_file);
	fclose(output_file);

	return 0;
}

//int main(int argc, char* argv[]) { return process_sequential(argc, argv); }
