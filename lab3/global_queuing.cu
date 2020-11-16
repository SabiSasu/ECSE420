/*
* ECSE420 LAB3: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
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

int read_input_one_two_four1(int** input1, char* filepath) {
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

int read_input_three1(int** input1, int** input2, int** input3, int** input4, char* filepath) {
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

__device__ int count = 0;

__global__ void global_queuing_kernel(int threadNum, int numCurrLevelNodes, int * numNextLevelNodes_h, 
									int * currLevelNodes_h, int* nodePtrs_h, int * nodeNeighbors_h, int * nodeVisited_h, 
									int * nodeGate_h, int * nodeInput_h, int * nodeOutput_h, int * nextLevelNodes_h, int blocknum){
	int i = threadIdx.x + (blockIdx.x * blockDim.x);
	int stride = blockDim.x * gridDim.x;
	
	// Loop over all nodes in the current level

	for (int idx = i; idx < numCurrLevelNodes; idx+= stride) {
		int node = currLevelNodes_h[idx];
		// Loop over all neighbors of the node
		for (int nbrIdx = nodePtrs_h[node]; nbrIdx < nodePtrs_h[node + 1]; nbrIdx++) {
			int neighbor = nodeNeighbors_h[nbrIdx];
			// If the neighbor hasn't been visited yet
			if (!nodeVisited_h[neighbor]) {
				// Mark it and add it to the queue
				nodeVisited_h[neighbor] = 1;

				//solve gate
				int result = 0;
				int output = nodeOutput_h[node];
				int input = nodeInput_h[neighbor];
				switch (nodeGate_h[neighbor]) {
					case AND: result = output & input;  break;
					case OR: result = output | input; break;
					case NAND: result = !(output & input); break;
					case NOR: result = !(output | input); break;
					case XOR: result = ((!output & input) | (output & !input)); break;
					case XNOR: result = !((!output & input) | (output & !input)); break;
				}

				nodeOutput_h[neighbor] = result;
				*numNextLevelNodes_h = count + 2 + threadNum - blocknum;
				nextLevelNodes_h[atomicAdd(&count, 1)] = neighbor;
				//printf("count is %d\n", *numNextLevelNodes_h);
			
				
				//printf("count is %d\n", *numNextLevelNodes_h);
			}
		}
	}
}


int process_global(int argc, char* argv[]) {

	//if (argc != 7)
	//	return 0;
	// get arguments from command line
	char* input_filename1 = "input1.raw";//argv[1];
	char* input_filename2 = "input2.raw";//argv[2];
	char* input_filename3 = "input3.raw";//argv[3];
	char* input_filename4 = "input4.raw";//argv[4];
	char* output_node_filename = "output/output_node.raw";//argv[5];
	char* output_next_node_filename = "output/output_next_node.raw";//argv[6];

	int mode = 1;
	//number of threads
	int num_of_threads = 0;
	int num_of_blocks = 0;

	if (mode == 1) { 
		num_of_threads = 32;
		num_of_blocks = 25;
	}
	else if (mode == 2) {
		num_of_threads = 64;
		num_of_blocks = 10;
	}
	else if (mode == 3) { 
		num_of_threads = 25;
		num_of_blocks = 64;
	}
	else if (mode == 4) { 
		num_of_threads = 35;
		num_of_blocks = 64;
	}
	else if (mode == 5) { 
		num_of_threads = 35;
		num_of_blocks = 128;
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
	int* numNextLevelNodes_h;
	int* nodeGate_h;
	int* nodeInput_h;
	int* nodeOutput_h;

	numNodePtrs = read_input_one_two_four1(&nodePtrs_h, input_filename1);
	numTotalNeighbors_h = read_input_one_two_four1(&nodeNeighbors_h, input_filename2);
	numNodes = read_input_three1(&nodeVisited_h, &nodeGate_h, &nodeInput_h, &nodeOutput_h, input_filename3);
	numCurrLevelNodes = read_input_one_two_four1(&currLevelNodes_h, input_filename4);

	//output
	int* nextLevelNodes_h = (int*)malloc(numNodePtrs * sizeof(int));

	//initialize cuda vars
	int* currLevelNodes_c, int* nodePtrs_c, int* nodeNeighbors_c, int* nodeVisited_c,
		int* nodeGate_c, int* nodeInput_c, int* nodeOutput_c, int* nextLevelNodes_c;
	cudaMalloc(&currLevelNodes_c, numCurrLevelNodes * sizeof(int));
	cudaMalloc(&nodePtrs_c, numNodePtrs * sizeof(int));
	cudaMalloc(&nodeNeighbors_c, numTotalNeighbors_h * sizeof(int));
	cudaMalloc(&nodeVisited_c, numNodes * sizeof(int));
	cudaMalloc(&nodeGate_c, numNodes * sizeof(int));
	cudaMalloc(&nodeInput_c, numNodes * sizeof(int));
	cudaMalloc(&nodeOutput_c, numNodes * sizeof(int));
	cudaMalloc(&nextLevelNodes_c, 40101 * sizeof(int));

	cudaMemcpy(currLevelNodes_c, currLevelNodes_h, numCurrLevelNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nodePtrs_c, nodePtrs_h, numNodePtrs * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nodeNeighbors_c, nodeNeighbors_h, numTotalNeighbors_h * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nodeVisited_c, nodeVisited_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nodeGate_c, nodeGate_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nodeInput_c, nodeInput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nodeOutput_c, nodeOutput_h, numNodes * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(nextLevelNodes_c, nextLevelNodes_h, 40101 * sizeof(int), cudaMemcpyHostToDevice);

	cudaMallocManaged(&numNextLevelNodes_h, sizeof(int));
	*numNextLevelNodes_h = 0;


	int numCurrLevelNodesPerThread = (numCurrLevelNodes - 1) / (num_of_blocks * num_of_threads);

	//start timer for execution runtime 
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);


	global_queuing_kernel << < num_of_blocks, num_of_threads >> > (num_of_threads, numCurrLevelNodes, numNextLevelNodes_h,
			currLevelNodes_c, nodePtrs_c, nodeNeighbors_c, nodeVisited_c, nodeGate_c, nodeInput_c, nodeOutput_c, nextLevelNodes_c, num_of_blocks);
	cudaDeviceSynchronize();

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Global queueing: ran in %f milliseconds\n", memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);


	//free cuda memory
	cudaMemcpy(nodeOutput_h, nodeOutput_c, (numNodePtrs-1) * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(nextLevelNodes_h, nextLevelNodes_c, *numNextLevelNodes_h * sizeof(int), cudaMemcpyDeviceToHost);
	
	cudaFree(currLevelNodes_c);
	cudaFree(nodePtrs_c);
	cudaFree(nodeNeighbors_c);
	cudaFree(nodeVisited_c);
	cudaFree(nodeGate_c);
	cudaFree(nodeInput_c);
	cudaFree(nodeOutput_c);
	cudaFree(nextLevelNodes_c);


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

	

	fprintf(output_file_node, "%d\n", numNodePtrs - 1);
	for (int loop = 0; loop < numNodePtrs - 1; loop++)
		fprintf(output_file_node, "%d\n", nodeOutput_h[loop]);
	fclose(output_file_node);

	fprintf(output_file_next, "%d\n", *numNextLevelNodes_h);
	for (int loop = 0; loop < *numNextLevelNodes_h; loop++)
		fprintf(output_file_next, "%d\n", nextLevelNodes_h[loop]);
	fclose(output_file_next);
	
	return 0;
}

int main(int argc, char* argv[]) { return process_global(argc, argv); }
