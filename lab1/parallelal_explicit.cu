/*
* ECSE420 LAB0: Group 15, Sabina Sasu & Erica De Petrillo
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

void operation(int file_length, FILE* input_file, FILE* output_file, int num_threads_per_block,) {

	//read through the whole document until u get to ur threadId + blockID * M number and then u do the thing fr tht line
	char line[256];
	int i = 0;
	int stop_at = threadIdx.x + blockIdx.x * num_threads_per_block;

	fgets(line, sizeof(line), input_file);

	while(i < file_length) {
		if (i == stop_at) {
			char gateA = line[0];
			char gateB = line[2];
			char gateNum = line[4];
			int* output = 0;

			switch (gateNum) {
			case AND: gateA& gateB;  break;
			case OR: gateA | gateB; break;
			case NAND: !(gateA & gateB); break;
			case NOR: !(gateA | gateB); break;
			case XOR: ((!gateA & gateB) | (gateA & !gateB)); break;
			case XNOR: !((!gateA & gateB) | (gateA & !gateB)); break;
			}

			fwrite(output, sizeof(int), 1, output_file);
			//fputs(output, output_file);
			break;
		}
		fgets(line, sizeof(line), input_file);
		i++;
	}


	fclose(input_file);
	fclose(output_file);
}


int process_parallel_explicit(int argc, char* argv[]) {

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

	int num_blocks = 0;
	int num_threads_per_block = 0;

	if (file_length <= 1024) {
		num_blocks = 1;
		num_threads_per_block = file_length
	}
	else {
		num_blocks = ((file_length - 1) / 1024) + 1; //1024 is the max number of threads in 1 block
		num_threads_per_block = ceil(file_length / num_blocks);
	}

	//would need to call the function

	return 0;
}

//int main(int argc, char* argv[]) { return process_sequential(argc, argv); }
