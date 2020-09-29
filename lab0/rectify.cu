/*
* ECSE420 LAB0: Group 15, Sabina Sasu & Erica Depatrillo
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void rectification(unsigned char* image, unsigned char* new_image, unsigned int size)
{
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        //if image[index] has a value lower than 127, then new_image[index] is 127
        new_image[index] = image[index] < 127 ? 127 : image[index];
    }
}

int process_rectify(int argc, char* argv[]) {

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
    if (error) 
        printf("error %u: %s\n", error, lodepng_error_text(error));
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

    //rectify
    rectification << < size/threadNum, threadNum >> > (image_cuda, new_image_rec_cuda, size);
    cudaDeviceSynchronize();

    //stop timer
    cudaEventRecord(stop, 0); cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop);
    printf("Rectify: thread count is %d, ran in %f milliseconds\n", threadNum, memsettime);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    //free cuda memory
    cudaMemcpy(new_image_rec, new_image_rec_cuda, size, cudaMemcpyDeviceToHost);
    cudaFree(image_cuda);
    cudaFree(new_image_rec_cuda);

    //save png image
    lodepng_encode32_file(output_filename, new_image_rec, width, height);

    //free memory
    free(image);
    free(new_image_rec);

    return 0;
}

int main(int argc, char* argv[]){   return process_rectify(argc, argv);}


