
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "lodepng.h"

#include <stdio.h>
#include <stdlib.h>

__global__ void process(char* input_filename, char* output_filename)
{
    unsigned error;
    unsigned char* image, * new_image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);

}

int main()
{
    char* input_filename = "Test_1.png";
    char* output_filename = "Test_1_output.png";

    process(input_filename, output_filename);

    return 0;
}
