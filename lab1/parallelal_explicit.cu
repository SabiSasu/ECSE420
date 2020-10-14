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



	char line[256];
	int i = 0;
	while (i < file_length) {
		fgets(line, sizeof(line), input_file);
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
	}


	fclose(input_file);
	fclose(output_file);

	return 0;
}

//int main(int argc, char* argv[]) { return process_sequential(argc, argv); }
