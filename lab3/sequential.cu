/*
* ECSE420 LAB3: Group 15, Sabina Sasu & Erica De Petrillo
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define AND 0
#define OR 1
#define NAND 2
#define NOR 3
#define XOR 4
#define XNOR 5

int read_input_one_two_four(int** input1, char* filepath) {
	FILE* fp = fopen(filepath, "r");
	if (fp == NULL) {
		fprintf(stderr, "Couldn't open file for reading\n");
		exit(1);
	}
	int counter = 0;
	int len;
	int length = fscanf(fp, "%d", &len);
	*input1 = (int*)malloc(len * sizeof(int));
	int temp1;
	while (fscanf(fp, "%d", &temp1) == 1) {
		(*input1)[counter] = temp1;

		counter++;
	}
	fclose(fp);
	return len;
}

int read_input_three(int** input1, int** input2, int** input3, int** input4, char* filepath) {
	FILE* fp = fopen(filepath, "r");
	if (fp == NULL) {
		fprintf(stderr, "Couldn't open file for reading\n");
		exit(1);
	}
	int counter = 0;
	int len;
	int length = fscanf(fp, "%d", &len);
	*input1 = (int*)malloc(len * sizeof(int));
	*input2 = (int*)malloc(len * sizeof(int));
	*input3 = (int*)malloc(len * sizeof(int));
	*input4 = (int*)malloc(len * sizeof(int));
	int temp1;
	int temp2;
	int temp3;
	int temp4;
	while (fscanf(fp, "%d,%d,%d,%d", &temp1, &temp2, &temp3, &temp4) == 4) {
		(*input1)[counter] = temp1;
		(*input2)[counter] = temp2;
		(*input3)[counter] = temp3;
		(*input4)[counter] = temp4;
		counter++;
	}
	fclose(fp);
	return len;
}

int gate_solver(int gate, int output, int input) {
	int result = 0;
	switch (gate) {
		case AND: result = output & input;  break;
		case OR: result = output | input; break;
		case NAND: result = !(output & input); break;
		case NOR: result = !(output | input); break;
		case XOR: result = ((!output & input) | (output & !input)); break;
		case XNOR: result = !((!output & input) | (output & !input)); break;
	}
	return result;
}

int process_sequential(int argc, char* argv[]) {

	char* input_filename1 = "input1.raw";//argv[1];
	char* input_filename2 = "input2.raw";//argv[2];
	char* input_filename3 = "input3.raw";//argv[3];
	char* input_filename4 = "input4.raw";//argv[4];
	char* output_node_filename = "output/output_node.raw";//argv[5];
	char* output_next_node_filename = "output/output_next_node.raw";//argv[6];

	//if we have the arguments from cmd, take them instead
	if (argc == 7) {
		char* input_filename1 = argv[1];
		char* input_filename2 = argv[2];
		char* input_filename3 = argv[3];
		char* input_filename4 = argv[4];
		char* output_node_filename = argv[5];
		char* output_next_node_filename = argv[6];
	}
																	
	//Code provided:
	//Variables
	int numNodePtrs;
	int numNodes;
	int* nodePtrs_h;
	int* nodeNeighbors_h;
	int* nodeVisited_h;
	int numTotalNeighbors_h;
	int* currLevelNodes_h;
	int numCurrLevelNodes;
	int numNextLevelNodes_h = 0;
	int* nodeGate_h;
	int* nodeInput_h;
	int* nodeOutput_h;

	numNodePtrs = read_input_one_two_four(&nodePtrs_h, input_filename1);
	numTotalNeighbors_h = read_input_one_two_four(&nodeNeighbors_h, input_filename2);
	numNodes = read_input_three(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, input_filename3);
	numCurrLevelNodes = read_input_one_two_four(&currLevelNodes_h, input_filename4);


	//output
	int* nextLevelNodes_h = (int*)malloc(numNodePtrs * sizeof(int));

	//start clock
	clock_t begin = clock();

	// Loop over all nodes in the current level
	for (int idx = 0; idx < numCurrLevelNodes; idx++) {
		int node = currLevelNodes_h[idx];
		// Loop over all neighbors of the node
		for (int nbrIdx = nodePtrs_h[node]; nbrIdx < nodePtrs_h[node + 1]; nbrIdx++) {
			int neighbor = nodeNeighbors_h[nbrIdx];
			// If the neighbor hasn't been visited yet
			if (!nodeVisited_h[neighbor]) {
				// Mark it and add it to the queue
				nodeVisited_h[neighbor] = 1;
				nodeOutput_h[neighbor] = gate_solver(nodeGate_h[neighbor], nodeOutput_h[node], nodeInput_h[neighbor]);
				nextLevelNodes_h[numNextLevelNodes_h] = neighbor;
				++(numNextLevelNodes_h);
			}
		}
	}

	clock_t end = clock();

	float time_spent = ((double)end - begin) / CLOCKS_PER_SEC * 1000;
	printf("Execution time: %f milliseconds\n", time_spent);

	//output results to respective files

	FILE* output_file_node;
	FILE* output_file_next;
	if ((output_file_node = fopen(output_node_filename, "w")) == NULL) {
		printf("Error! opening file");
		exit(1);
	}
	if ((output_file_next = fopen(output_next_node_filename, "w")) == NULL) {
		printf("Error! opening file");
		exit(1);
	}
	//first line is the length
	fprintf(output_file_node, "%d\n", numNodePtrs-1);
	for (int loop = 0; loop < numNodePtrs-1; loop++)
		fprintf(output_file_node, "%d\n", nodeOutput_h[loop]);
	fclose(output_file_node);

	fprintf(output_file_next, "%d\n", numNextLevelNodes_h);
	for(int loop = 0; loop < numNextLevelNodes_h-1; loop++)
		fprintf(output_file_next, "%d\n", nextLevelNodes_h[loop]);

	
	fclose(output_file_next);

	return 0;
}

//int main(int argc, char* argv[]) { return process_sequential(argc, argv); }
