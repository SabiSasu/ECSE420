
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define max(a,b) ((a) > (b) ? (a) : (b))

__global__ void pooling(unsigned char* image, unsigned char* new_image, unsigned width, unsigned int size, unsigned int blocks_per_row)
{
	unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    //unsigned int index = threadIdx.x + (blockIdx.x % blocks_per_row) * blockDim.x;
	unsigned int new_index = index / 4; //index /2;

    if (index < size) {
        //loop through rgba
        for (int k = 0; k < 4; k++) {
            int tl = (int)image[index + k]; //top left
            int tr = (int)image[index + 4 + k]; //top right
            int bl = (int)image[index + (4*width) + k]; //bot left
            int br = (int)image[index + (4 * width) + 4 + k]; //bot right
           
            signed int val = max(max(tl, bl), max(tr, br));
                
            //assign new value to pixel
            new_image[new_index + k] = (unsigned char)val;
        }
    }
}

__global__ void compression(unsigned char* image, unsigned char* new_image, unsigned width, unsigned int size, unsigned int blocks_per_row) {
	unsigned int index = threadIdx.x + (blockIdx.x % blocks_per_row) * blockDim.x;
	unsigned int new_index = (threadIdx.x + blockIdx.x * blockDim.x) + 4;

	if (index < size) {
		for (int i = 0; i < 4; i++) {						// iterate through R, G, B, A
			unsigned int max = image[index];
			if (image[index + 4 + i] > max) {				// pixel to the right
				max = image[index + 4 + i];
			}
			if (image[index + (4 * width) + i] > max) {		// pixel below
				max = image[index + (4 * width) + i];
			}
			if (image[index + (4 * width) + 4 + i] > max) {	// pixel below & to the right
				max = image[index + (4 * width) + 4 + i];
			}
			new_image[new_index + i] = max;
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
    unsigned int num_blocks = ceil((size / threadNum) / 16) + 1;
    unsigned int blocks_per_row = ceil(width / threadNum);

    // pool
    compression << < num_blocks, threadNum >> > (image_cuda, new_image_rec_cuda, width, size, blocks_per_row);
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

int process_pool2() {
	// file definitions
	char* filename1 = "Test Images\\Test_1.png";						// change these depending on what we're doing
	//char* filename2 = "test_rectify_result.png";		// output for rectify
	char* filename3 = "Output Images\\Test_1_output_pool.png";		// output for pooling
	//char* filename4 = "test_rectify_expected_result.png";	// filename for rectify comparison
	//char* filename5 = "test_pooling_expected_result.png";	// filename for pooling comparison

	// load input image
	unsigned char* image;
	unsigned width, height;
	unsigned error = lodepng_decode32_file(&image, &width, &height, filename1);
	if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
	unsigned int size_image = width * height * 4 * sizeof(unsigned char); // height x width number of pixels, 4 layers (RGBA) for each pixel, 1 char for each value

	// define number of threads
	unsigned int thread_number = 256;		// number of threads per block we're using
	unsigned int thread_max = 1024;			// hardware limit: maximum number of threads per block

	if (thread_number > thread_max) {		// can't have more threads than the hardware limit
		thread_number = thread_max;
	}


	/********** Pooling Start ***********/
	// allocate memory space on GPU
	unsigned char* cuda_image_pool, * cuda_new_image_pool;
	cudaMalloc((void**)&cuda_image_pool, size_image);
	cudaMalloc((void**)&cuda_new_image_pool, size_image);

	// CPU copies input data from CPU to GPU
	cudaMemcpy(cuda_image_pool, image, size_image, cudaMemcpyHostToDevice);

	// maximum number of threads we can use is 1 per 16 pixel values
		// that's because we can use maximum 1 thread per 2x2 area, and each pixel in that 2x2 area has 4 values
	if (thread_number > ceil(size_image / 16)) {
		thread_number = ceil(size_image / 16);
	}

	// figure out how many blocks we need for this task
	unsigned int num_blocks = 36000 * 4;//144000;//ceil((size_image / thread_number)/*/16*/) /*+ 1*/;
	unsigned int blocks_per_row =7200; //28800;//ceil(width / thread_number);
	//unsigned int num_blocks = blocks_per_row * height / 2;

	// call method on GPU
	pooling << < num_blocks, thread_number >> > (cuda_image_pool, cuda_new_image_pool, width, size_image, blocks_per_row);
	//compression << <1, 1 >> > (cuda_image_pool, cuda_new_image_pool, width, size_image, 1);
	cudaDeviceSynchronize();

	// CPU copies input data from GPU back to CPU
	unsigned char* new_image_pool = (unsigned char*)malloc(size_image);
	cudaMemcpy(new_image_pool, cuda_new_image_pool, size_image, cudaMemcpyDeviceToHost);
	cudaFree(cuda_image_pool);
	cudaFree(cuda_new_image_pool);

	lodepng_encode32_file(filename3, new_image_pool, width / 2, height / 2);
	/********** Pooling End ***********/



	free(image);
	free(new_image_pool);
	return 0;
}

int main(int argc, char* argv[]){return process_pool2();}
