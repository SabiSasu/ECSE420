/*
* ECSE420 LAB3: Group 15, Sabina Sasu & Erica De Petrillo
*/


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


int process_sequential(int argc, char* argv[]) {

	if (argc != 7)
		return 0;

	// get arguments from command line
	char* input_filename1 = "input1.raw";//argv[1];
	char* input_filename2 = "input2.raw";//argv[2];
	char* input_filename3 = "input3.raw";//argv[3];
	char* input_filename4 = "input4.raw";//argv[4];
	char* output_node_filename = "output/output_node.raw";//argv[5];
	char* output_next_node_filename = "output/output_next_node.raw";//argv[6];


	float u2[4][4] = { //previous previous array [rows] [columns]
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0}
	};

	float u1[4][4] = { //previous array [rows] [columns]
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0}
	};
	int n = 4; //array is n x n

	u1[n / 2][n / 2] = 1; //simulated hit on the drum

	float u[4][4] = { //array we will work on
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0},
		{0, 0, 0, 0}
	};
	float p = 0.5;
	float eta = 0.0002;
	float g = 0.75; //boundary gain

	for (int t = 0; t < num_of_iterations; t++) {

		//update all the interior elements
		for (int i = 1; i < (n - 1); i++) { //avoid first and last row
			for (int j = 1; j < (n - 1); j++) { //avoid first and last column
				u[i][j] = (p * (u1[i - 1][j] + u1[i + 1][j] + u1[i][j - 1] + u1[i][j + 1] - 4 * u1[i][j]) + 2 * u1[i][j] - (1 - eta) * u2[i][j])
					/ (1 + eta);
			}
		}

		//update boundaries
		for (int k = 1; k < (n - 1); k++) {
			u[0][k] = g * u[1][k];
			u[n - 1][k] = g * u[n - 2][k];
			u[k][0] = g * u[k][1];
			u[k][n - 1] = g * u[k][n - 2];
		}

		//update corners
		u[0][0] = g * u[1][0];
		u[n - 1][0] = g * u[n - 2][0];
		u[0][n - 1] = g * u[0][n - 2];
		u[n - 1][n - 1] = g * u[n - 1][n - 2];

		//update u1 and u2
		memcpy(u2, u1, sizeof(u1));
		memcpy(u1, u, sizeof(u1));

		//print out position u[n/2][n/2]
		printf("iteration %d: position (%d, %d) = %f \n", t, n / 2, n / 2, u[n / 2][n / 2]);

		//printing full matrix but only for debugging purposes
		/*printf("iteration %d: \n", t);
		printf("%f %f %f %f \n", u[0][0], u[0][1], u[0][2], u[0][3]);
		printf("%f %f %f %f \n", u[1][0], u[1][1], u[1][2], u[1][3]);
		printf("%f %f %f %f \n", u[2][0], u[2][1], u[2][2], u[2][3]);
		printf("%f %f %f %f \n", u[3][0], u[3][1], u[3][2], u[3][3]);*/

	}


	return 0;
}

int main(int argc, char* argv[]) { return process_sequential(argc, argv); }
