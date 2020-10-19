/*
* ECSE420 LAB1: Group 15, Sabina Sasu & Erica De Petrillo
*/


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
	
	clock_t end = clock();

	float time_spent = ((double)end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Execution time: %f milliseconds\n",  time_spent);

	fclose(input_file);
	fclose(output_file);

	return 0;
}

int main(int argc, char* argv[]) { return process_sequential(argc, argv); }
