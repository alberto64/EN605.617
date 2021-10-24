#include <stdio.h>
#include <cublas.h>
#include <cublas_v2.h>
#define indexCalculation(i,j,ld) (((j)*(ld))+(i))

/**
* printMatrix: A method that takes in a matrix and its dimentions and it prints
*/
void printMatrix(const char* name, float *matrix, int matrixWidth, int matrixHeight) {
	printf("\n%s: [ ", name);
	for(int i = 0 ; i < matrixHeight ; i++) {
		printf("\n");
		for(int j = 0 ; j < matrixWidth ; j++) {
			printf("%f, ", matrix[indexCalculation(i, j, matrixHeight)]);
		}
	}
	printf("]");
}

/**
* runOperation: Taking the number of blocks and threads it does an operation on the two 
* given matrices and prints their results.
*/
void runOperation(int matrixHeight, int matrixWidth) { 
    
	// Setup Timing Variables
	cudaEvent_t start, stop; 
	float elapsedTimeInMiliseconds; 
	cudaEventCreate(&start); 
	cudaEventCreate(&stop); 

	// Setup host memory variables
    cublasInit();
	
	// Setup Handle and stream
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaStream_t operationStream; 
	cudaStreamCreate(&operationStream);
	cublasSetStream(handle, operationStream);

    float *mA = (float*) malloc(matrixHeight * matrixWidth * sizeof(float));
    float *mB = (float*) malloc(matrixHeight * matrixWidth * sizeof(float));
    float *mC = (float*) malloc(matrixHeight * matrixWidth * sizeof(float));

    for (int i = 0 ; i < matrixHeight ; i++) {
      	for (int j = 0 ; j < matrixWidth ; j++) {
        	mA[indexCalculation(i,j,matrixHeight)] = (float) indexCalculation(i,j,matrixHeight);
			mB[indexCalculation(i,j,matrixHeight)] = (float) indexCalculation(i,j,matrixHeight); 
		}   
	}
    
	// Turned off to minimize printing
	printMatrix("Matrix A", mA, matrixWidth, matrixHeight);
	printMatrix("Matrix B", mB, matrixWidth, matrixHeight);

	// Setup device memory variables
	float* dev_mA; float* dev_mB; float* dev_mC;
	cublasAlloc(matrixHeight * matrixWidth, sizeof(float), (void**) &dev_mA);
	cublasAlloc(matrixHeight * matrixWidth, sizeof(float), (void**) &dev_mB);
	cublasAlloc(matrixHeight * matrixWidth, sizeof(float), (void**) &dev_mC);

	cudaEventRecord(start);

	// Copy matrix from host to device memory
	cublasSetMatrix(matrixHeight, matrixWidth, sizeof(float), mA, matrixHeight, dev_mA, matrixHeight);
	cublasSetMatrix(matrixHeight, matrixWidth, sizeof(float), mB, matrixHeight, dev_mB, matrixHeight);

    // Execute Multiplication
	const float alf = 1;
	const float bet = 0;
	const float *alpha = &alf;
	const float *beta = &bet;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, matrixHeight, matrixWidth, matrixWidth, alpha, dev_mA, matrixHeight, dev_mB, matrixHeight, beta, dev_mC, matrixHeight);

	// Get result
    cublasGetMatrix(matrixHeight, matrixWidth, sizeof(float), dev_mC, matrixHeight, mC, matrixHeight);

    // Timing Output
	cudaStreamSynchronize(operationStream);
	cublasDestroy(handle);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&elapsedTimeInMiliseconds, start, stop); 

  	// Turned off to minimize printing
	printMatrix("Matrix C", mC, matrixWidth, matrixHeight);
	printf("\nStream and Event Time: %f Miliseconds\n", elapsedTimeInMiliseconds) * 100;

	// Free reserved memory
    free(mA); 
	free(mB);
	free(mC);
	cublasFree(dev_mA);
	cublasFree(dev_mB);
	cublasFree(dev_mC);
	cublasShutdown();
}

int main(int argc, char** argv) {
	int matrixHeight = 1024;

	// read command line arguments
	if (argc >= 2) {
		matrixHeight = atoi(argv[1]);
	}

	printf("\nMatrix Width and Height: %d\n", matrixHeight);
	
	runOperation(matrixHeight, matrixHeight);

	return 0;
}
