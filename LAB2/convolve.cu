/*
* ECSE420 LAB2: Group 15, Sabina Sasu & Erica De Petrillo
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "wm.h"
#include "lodepng.h"

#define weight_matrix wm.w //Default matrix size NxN

__global__ void convolution(unsigned char* image, unsigned char* new_image, unsigned width, unsigned int newSize, unsigned int threadNum)
{
	float w[3][3] =
	{
	  1,	2,		-1,
	  2,	0.25,	-2,
	  1,	-2,		-1
	};

	
	for (int i = threadIdx.x; i < newSize; i += threadNum) {

		//to get the correct index of the top left pixel, need to offset
		//to make sure that we don't go all the way to the last 2 pixels on each row (8 indices)
		unsigned int offset = (2 * 4 * ((i/4) / ((width-2))));
		double value = 0.0;
		
		//skip alpha value, which is a modulo of 4
		if (i == 0 || (i + offset + 1) % 4 != 0) {
			//top row
			value += ((int)(image[i + offset]) * w[0][0]);
			value += ((int)(image[i + offset + 4]) * w[0][1]);
			value += ((int)(image[i + offset + 8]) * w[0][2]);
			//middle row
			value += ((int)(image[i + offset + (4 * width)]) * w[1][0]);
			value += ((int)(image[i + offset + (4 * width) + 4]) * w[1][1]);
			value += ((int)(image[i + offset + (4 * width) + 8]) * w[1][2]);
			//bottom row
			value += ((int)(image[i + offset + (8 * width)]) * w[2][0]);
			value += ((int)(image[i + offset + (8 * width) + 4]) * w[2][1]);
			value += ((int)(image[i + offset + (8 * width) + 8]) * w[2][2]);

			if (value < 0)
				value = 0;
			if (value > 255)
				value = 255;
		}
		else
			value = (int)image[i + offset];
		//assign new value to pixel
		new_image[i] = (unsigned int)round(value);
	}

}

int process_convolve(int argc, char* argv[])
{
	if (argc != 4)
		return 0;
	
	// get arguments from command line
	char* input_filename = argv[1];
	char* output_filename = argv[2];
	int threadNum = atoi(argv[3]);

    if (threadNum < 0 || threadNum > 1024)
        return 0;

	//getting image and its size
	unsigned error;
	unsigned char* image, * new_image_rec;
	unsigned width, height;

	error = lodepng_decode32_file(&image, &width, &height, input_filename);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	unsigned int size = width * height * 4 * sizeof(unsigned char);
	unsigned int newSize = (width - 2) * (height - 2) * 4 * sizeof(unsigned char);
	new_image_rec = (unsigned char*)malloc(newSize);
	


	//defining device vars
	unsigned char* image_cuda, * new_image_rec_cuda;
	cudaMalloc((void**)&image_cuda, size);
	cudaMalloc((void**)&new_image_rec_cuda, newSize);
	cudaMemcpy(image_cuda, image, size, cudaMemcpyHostToDevice);

	//start timer
	float memsettime;
	cudaEvent_t start, stop;
	cudaEventCreate(&start); cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

    //rectify
    convolution << < 1, threadNum >> > (image_cuda, new_image_rec_cuda, width, newSize, threadNum);
	cudaDeviceSynchronize();

	//stop timer
	cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
	cudaEventElapsedTime(&memsettime, start, stop);
	printf("\nPool: thread count is %d, ran in %f milliseconds\n", threadNum, memsettime);
	cudaEventDestroy(start); cudaEventDestroy(stop);

	//free cuda memory
	cudaMemcpy(new_image_rec, new_image_rec_cuda, newSize, cudaMemcpyDeviceToHost);
	cudaFree(image_cuda);
	cudaFree(new_image_rec_cuda);

	//save png image
	lodepng_encode32_file(output_filename, new_image_rec, width-2, height-2);
	//free memory
	free(image);
	free(new_image_rec);

	return 0;
}

int main(int argc, char* argv[]) { return process_convolve(argc, argv); }
