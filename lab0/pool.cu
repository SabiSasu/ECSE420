
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


/*
int main(int argc, char* argv[])
{
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