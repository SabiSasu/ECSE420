/*
* ECSE420 LAB0: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define max(a,b) ((a) > (b) ? (a) : (b))

__global__ void pooling(unsigned char* image, unsigned char* new_image, unsigned width, unsigned int size)
{
	unsigned int new_index = threadIdx.x + blockIdx.x * blockDim.x;
	//multiplied by 8 to account for the 2 pixels in width of each 2x2 matrix multiplied by 4 RGBA values
	unsigned int index = new_index * 8;

	if (index < size) {
		//4 * width because of RGBA values
		//(width / 2) --> the number of 2x2 matrices in 1 row
		//new_index / (width / 2) --> see which row we're in
		//to get the correct index of the top left pixel in each 2x2 matrix,
		//we need to add this constant
		int c = (4 * width) * (new_index / (width / 2));

		//loop through rgba
		for (int k = 0; k < 4; k++) {
			int tl = (int)image[index + c + k]; //top left
			int tr = (int)image[index + c + 4 + k]; //top right
			int bl = (int)image[index + c + (4 * width) + k]; //bot left
			int br = (int)image[index + c + (4 * width) + 4 + k]; //bot right
			//get max
			signed int val = max(max(tl, bl), max(tr, br));

			//assign new value to pixel
			new_image[new_index * 4 + k] = (unsigned char)val;
		}
	}
}

int process_pool(int argc, char* argv[]) {

	if (argc != 4)
		return 0;

	// get arguments from command line
	char* input_filename = argv[1];
	char* output_filename = argv[2];
	int threadNum = atoi(argv[3]);

	//getting image and its size
	unsigned error;
	unsigned char* image, * new_image_rec;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	unsigned int size = width * height * 4 * sizeof(unsigned char);
	new_image_rec = (unsigned char*)malloc(size);

	//defining device vars
	unsigned char* image_cuda, * new_image_rec_cuda;
	cudaMalloc((void**)&image_cuda, size);
	cudaMalloc((void**)&new_image_rec_cuda, size);
	cudaMemcpy(image_cuda, image, size, cudaMemcpyHostToDevice);

	//start timer
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	// figure out how many blocks we need for this task
	unsigned int blocks = width * height / 4 / threadNum;

	// pool
	pooling << < blocks, threadNum >> > (image_cuda, new_image_rec_cuda, width, size);
	cudaDeviceSynchronize();

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("Pool: thread count is %d, ran in %f milliseconds\n", threadNum, memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	//free cuda memory
	cudaMemcpy(new_image_rec, new_image_rec_cuda, size, cudaMemcpyDeviceToHost);
	cudaFree(image_cuda);
	cudaFree(new_image_rec_cuda);

	//save png image
	lodepng_encode32_file(output_filename, new_image_rec, width / 2, height / 2);

	//free memory
	free(image);
	free(new_image_rec);

	return 0;
}

int main(int argc, char* argv[]) { return process_pool(argc, argv); }
