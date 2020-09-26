
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void rectification(unsigned char* image, unsigned char* new_image, unsigned int size)
{
    //change this so it doesnt look too conspicuous
    unsigned int index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < size) {
        
        if (image[index] < 127) {
            new_image[index] = 127;
        }
        else {
            new_image[index] = image[index];
        }
    }

}


int main(int argc, char* argv[])
{

    // get arguments from command line
    char* input_filename = "Test Images\\Test_1.png"; //argv[1];
    char* output_filename = "Output Images\\Test_1_output.png"; //argv[2];
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
    cudaMalloc((void**) &image_cuda, size);
    cudaMalloc((void**) &new_image_rec_cuda, size);
    cudaMemcpy(image_cuda, image, size, cudaMemcpyHostToDevice);

    //start timer
    float memsettime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    //rectify
    rectification << < (size + threadNum - 1) / threadNum, threadNum >> > (image_cuda, new_image_rec_cuda, size);
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


