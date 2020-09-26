#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "lodepng.h"
#include <stdio.h>
#include <stdlib.h>

void process(char* input_filename, char* output_filename)
{
    unsigned error;
    unsigned char* image, * new_image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, input_filename);
    if (error) printf("error %u: %s\n", error, lodepng_error_text(error));
    new_image = (unsigned char*)malloc(width * height * 4 * sizeof(unsigned char));


    //to make parallel
    unsigned char value;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {

            for (int k = 0; k < 3; k++) {
                signed int val = (int)image[4 * width * i + 4 * j + k];
                val -= 127;
                val = (val >= 0) ? val : 0;
                val += 127;
                new_image[4 * width * i + 4 * j + k] = (unsigned char)val;
            }

            new_image[4 * width * i + 4 * j + 3] = image[4 * width * i + 4 * j + 3]; //A
        }
    }


    //saves image and frees pointers
    lodepng_encode32_file(output_filename, new_image, width, height);

    free(image);
    free(new_image);
}

/*
int main()
{
    char* input_filename = "Test Images\\Test_1.png";
    char* output_filename = "Output Images\\Test_1_output.png";

    process(input_filename, output_filename);

    return 0;
}
*/