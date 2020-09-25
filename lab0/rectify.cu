
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void rectification(unsigned char* image, unsigned char* new_image, unsigned width, unsigned height)
{

    unsigned int i = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {

        for (int k = 0; k < 3; k++) {
            signed int val = image[4 * width * i + 4 * j + k];
            val -= 127;
            val = (val >= 0) ? val : 0;
            val += 127;
            new_image[4 * width * i + 4 * j + k] = (unsigned char)val;
        }

        //new_image[4 * width * i + 4 * j + 0] = image[4 * width * i + 4 * j + 0];
       // new_image[4 * width * i + 4 * j + 1] = image[4 * width * i + 4 * j + 1];
        //new_image[4 * width * i + 4 * j + 2] = image[4 * width * i + 4 * j + 2];
        new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3];
    }

}


int main(int argc, char* argv[])
{

        // get arguments from command line

        //char* input_filename = argv[1];
        //char* output_filename = argv[2];
       // int threadNum = atoi(argv[3]);
    int threadNum = 1;
    char* input_filename = "Test Images\\Test_1.png";

    unsigned error;
    unsigned char* image, *output, * new_image_rec;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    unsigned size = width * height;
    char snum[3];
    //for (int threads = 1; threads <= 256; threads * 2) {
    char* output_filename = "Output Images\\Test_1_output.png";
    //strcat(output_filename, itoa(threadNum, snum, 10));
    //strcat(output_filename, ".png");

    new_image_rec = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));

    cudaMalloc((void**)&output, width * height * 4 * sizeof(unsigned char));
    cudaMemcpy(output, new_image_rec, width * height * 4 * sizeof(unsigned char), cudaMemcpyHostToDevice);
    //timer
    time_t start, end;
    time(&start);

    // dim3 threads_per_block(width, height);
     //dim3 blocks_per_dimension(block_x_dim, block_y_dim);
     //rectificy
    rectification << <size/ threadNum, threadNum >> > (image, output, width, height);
    cudaDeviceSynchronize();

    //finish timer
    time(&end);
    printf("Rectify: thread count is %d, ran in %d seconds\n", threadNum, end - start);

    //saves image and frees pointers
    lodepng_encode32_file(output_filename, new_image_rec, width, height);

    cudaFree(output);
    free(image);
    free(new_image_rec);

    // }

    return 0;
}


