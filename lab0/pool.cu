
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define max(a,b) ((a) > (b) ? (a) : (b))

__global__ void pooling(unsigned char* image, unsigned char* new_image, unsigned width, unsigned height)
{
    for (int i = 0; i < height; i+=2) {
        for (int j = 0; j < width; j+=2) {

          //loop through rgba
            for (int k = 0; k < 4; k++) {
                int tl = (int)image[4*width*i + 4*j + k]; //top left
                int bt = (int)image[4*width*(i) + 4*(j + 1) + k]; //bot left
                int tr = (int)image[4*width*(i + 1) + 4*j + k]; //top right
                int br = (int)image[4*width*(i + 1) + 4*(j + 1) + k]; //bot right
           
                signed int val = max(max(tl, bt), max(tr, br));
                
                //assign new value to pixel
                new_image[4*width * (i/2) + 4*(j/2) + k] = (unsigned char)val;
            }
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

/*
int main(int argc, char* argv[])
{
    // allocate memory space on GPU
    unsigned char* cuda_image_pool, * cuda_new_image_pool;
    cudaMalloc((void**)& cuda_image_pool, size_image);
    cudaMalloc((void**)& cuda_new_image_pool, size_image);

    // CPU copies input data from CPU to GPU
    cudaMemcpy(cuda_image_pool, image, size_image, cudaMemcpyHostToDevice);

    // maximum number of threads we can use is 1 per 16 pixel values
        // that's because we can use maximum 1 thread per 2x2 area, and each pixel in that 2x2 area has 4 values
    if (thread_number > ceil(size_image / 16)) {
        thread_number = ceil(size_image / 16);
    }

    // figure out how many blocks we need for this task
    num_blocks = ceil((size_image / thread_number) / 16) + 1;
    unsigned int blocks_per_row = ceil(width / thread_number);

    // call method on GPU
    compression <<< num_blocks, thread_number >>> (cuda_image_pool, cuda_new_image_pool, width, size_image, blocks_per_row);
    cudaDeviceSynchronize();

    // CPU copies input data from GPU back to CPU
    unsigned char* new_image_pool = (unsigned char*)malloc(size_image);
    cudaMemcpy(new_image_pool, cuda_new_image_pool, size_image, cudaMemcpyDeviceToHost);
    cudaFree(cuda_image_pool);
    cudaFree(cuda_new_image_pool);

    lodepng_encode32_file(filename3, new_image_pool, width / 2, height / 2);
    // get arguments from command line
    if (argc < 4)
    {
        printf("Not enough arguments. Input arguments as follows:\n"
            "./pool <name of input png> <name of output png> <# threads>\n");
        return 0;
    }

    char* input_filename = argv[1];
    char* output_filename = argv[2];
    int threadNum = atoi(argv[3]);


    unsigned error;
    unsigned char* image,  * new_image_poo;
    unsigned width, height;


    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    
    unsigned size = width * height;
    //allocated half the space since image has half the pixels
    new_image_poo = (unsigned char*)malloc(width/2 * height/2 * 4 * sizeof(unsigned char));

    //timer
    time_t start, end;
    start = clock();

    pooling << <size / threadNum, threadNum >> > (image, new_image_poo, width, height);
    //finish timer
    end = clock();
    printf("Thread count is %s, ran in %s seconds\n", threadNum, end);
    //t = (end - start) / CLOCKS_PER_SEC;

    //saves image and frees pointers
    lodepng_encode32_file(output_filename, new_image_poo, width, height);

    free(image);
    free(new_image_poo);


    return end;
}
*/