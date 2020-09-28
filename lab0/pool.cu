

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
	unsigned int index = new_index * 8;

	if (index < size) {
		//do some witchcraft
		int c = (4 * width) * (new_index / (width / 2));

		//loop through rgba
		for (int k = 0; k < 4; k++) {
			int tl = (int)image[index + c + k]; //top left
			int tr = (int)image[index + c + 4 + k]; //top right
			int bl = (int)image[index + c + (4 * width) + k]; //bot left
			int br = (int)image[index + c + (4 * width) + 4 + k]; //bot right

			signed int val = max(max(tl, bl), max(tr, br));

			//assign new value to pixel
			new_image[new_index * 4 + k] = (unsigned char)val;
		}
	}
}

int process_pool() {
	// get arguments from command line
	char* input_filename = "Test Images\\Test_1.png"; //argv[1];
	char* output_filename = "Output Images\\Test_1_output_pool.png"; //argv[2];
	int threadNum = 1000; //atoi(argv[3]);

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
	unsigned int num_blocks = width * height / 4 / threadNum;

	// pool
	pooling << < num_blocks, threadNum >> > (image_cuda, new_image_rec_cuda, width, size);
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

int main(int argc, char* argv[]) { return process_pool(); }
