#include <stdio.h>
#include <time.h>
#define indexCalculation(i,j,ld) (((j)*(ld))+(i))

/**
* printMatrix: A method that takes in a matrix and its dimentions and it prints
*/
void printMatrix(const char* name, float *matrix, int matrixWidth, int matrixHeight) {
	printf("\n%s: [ ", name);
	for(int i = 0 ; i < matrixHeight ; i++) {
		printf("\n");
		for(int j = 0 ; j < matrixWidth ; j++) {
			printf("%f ,", matrix[indexCalculation(i, j, matrixHeight)]);
		}
	}
	printf(" ]", name);
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
  
	// Setup CUDA Streams
	cudaStream_t operationStream; 
	cudaStreamCreate(&operationStream); 
  
	// Setup host memory variables
    cublasInit();

    float *mA = (float*) malloc(matrixHeight * matrixWidth * sizeof(float));
    float *mB = (float*) malloc(matrixHeight * matrixWidth * sizeof(float));
    float *mC = (float*) malloc(matrixHeight * matrixWidth * sizeof(float));

    for (int i = 0 ; i < matrixHeight ; i++) {
      	for (int j = 0 ; j < matrixWidth ; j++) {
        	A[index(i,j,matrixHeight)] = (float) index(i,j,matrixHeight);
			B[index(i,j,matrixHeight)] = (float) index(i,j,matrixHeight); 
		}   
	}
    
	// Turned off to minimize printing
	printMatrix("Matrix A", mA, matrixWidth, matrixHeight);
	printMatrix("Matrix B", mB, matrixWidth, matrixHeight);

	// Start Stream
	cublasSetStream(operationStream);

	// Setup device memory variables
	float* dev_mA; float* dev_mB; float* dev_mC;
	cublasAlloc(matrixHeight * matrixWidth, sizeof(float), (void**) &dev_mA);
	cublasAlloc(matrixHeight * matrixWidth, sizeof(float), (void**) &dev_mB);
	cublasAlloc(matrixHeight * matrixWidth, sizeof(float), (void**) &dev_mC);

	// Copy matrix from host to device memory
	cublasSetMatrix(matrixHeight, matrixWidth, sizeof(float), mA, matrixHeight, dev_mA, matrixHeight);
	cublasSetMatrix(matrixHeight, matrixWidth, sizeof(float), mB, matrixHeight, dev_mB, matrixHeight);

    // Execute Multiplication
    cublasSgemm('n','n', matrixHeight, matrixWidth, matrixWidth,1 /* Alpha */, dev_mA, matrixHeight, dev_mB, matrixHeight, 0 /* Beta */, dev_mC, matrixHeight);

	// Get result
    cublasGetMatrix(matrixHeight, matrixWidth, sizeof(float), dev_mC, matrixHeight, mC, matrixHeight);

    // Timing Output
	cudaStreamSynchronize(operationStream);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&elapsedTimeInMiliseconds, start, stop); 
  	printf("Stream and Event Time: %f Miliseconds\n", elapsedTimeInMiliseconds) * 100;

  	// Turned off to minimize printing
	printMatrix("Matrix C", mC, matrixWidth, matrixHeight);

	// Free reserved memory
    free(mA); 
	free(mB);
	free(mC);
	cublasFree(dev_mA);
	cublasFree(dev_mB);
	cublasFree(dev_mC);
}

int main(int argc, char** argv) {
	int matrixHeight = 1024;

	// read command line arguments
	if (argc >= 2) {
		matrixHeight = atoi(argv[1]);
	}

	printf("\nMatrix Width and Height: %d\n", matrixHeight);
	
	runOperations(matrixHeight, matrixHeight);

	return 0;
}
