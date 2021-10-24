#include <stdio.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define indexCalculation(i,j,ld) (((j)*(ld))+(i))

/**
* printMatrix: A method that takes in a matrix and its dimentions and it prints
*/
void printMatrix(const char* name, double *matrix, int matrixWidth, int matrixHeight) {
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
void runOperation(int matrixHeight, int matrixWidth, int nrhs) { 
    
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
	cusolverDnHandle_t solver;
	cusolverDnCreate(&solver);
	cudaStream_t operationStream; 
	cudaStreamCreate(&operationStream);
	cublasSetStream(handle, operationStream);

    double *mA = (double*) malloc(matrixHeight * matrixWidth * sizeof(double));
    double *vB = (double*) malloc(matrixHeight * nrhs * sizeof(double));
    double *mX = (double*) malloc(matrixHeight * nrhs * sizeof(double));

    for (int i = 0 ; i < matrixHeight ; i++) {
      	for (int j = 0 ; j < matrixWidth ; j++) {
        	mA[indexCalculation(i,j,matrixHeight)] = (double) indexCalculation(i,j,matrixHeight);
		}   
		for (int j = 0 ; j < nrhs; j++) {
			vB[indexCalculation(i,j,matrixHeight)] = (double) indexCalculation(i,j,matrixHeight); 
		}   

	}
    
	// Turned off to minimize printing
	printMatrix("Matrix A", mA, matrixWidth, matrixHeight);
	printMatrix("Vector B", vB, nrhs, matrixHeight);

	// Setup device memory variables
	int* dev_Info; 
	int  lwork = 0; 
	double* dev_mA; double* dev_vB; double* dev_tau; double *dev_work;
	const double one = 1;

	cudaMalloc((void**) &dev_mA  , sizeof(double) * matrixHeight * matrixWidth);
    cudaMalloc((void**) &dev_tau, sizeof(double) * matrixHeight);
    cudaMalloc((void**) &dev_vB  , sizeof(double) * matrixHeight * nrhs);
    cudaMalloc((void**) &dev_Info, sizeof(int));

	cudaEventRecord(start);

    cudaMemcpy(dev_mA, mA, sizeof(double) * matrixHeight * matrixWidth, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_vB, vB, sizeof(double) * matrixHeight * nrhs, cudaMemcpyHostToDevice);
 
	// Query working space of geqrf and ormqr
    cusolverDnDgeqrf_bufferSize(solver, matrixHeight, matrixWidth, dev_mA, matrixHeight, &lwork);
	cudaMalloc((void**) &dev_Info, sizeof(double) * lwork);

	// Compute QR factorization
	cusolverDnDgeqrf(solver, matrixHeight, matrixWidth, dev_mA, matrixHeight, dev_tau, dev_work, lwork, dev_Info);
    cudaDeviceSynchronize();

	// Compute Q^T*B
    cusolverDnDormqr(solver, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, matrixHeight, nrhs, matrixWidth, dev_mA, matrixHeight,
        dev_tau, dev_vB, matrixHeight, dev_work, lwork, dev_Info);
    cudaDeviceSynchronize();

	// Compute x = R \ Q^T*B

    cublasDtrsm(handle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, matrixHeight,
        nrhs, &one, dev_mA, matrixHeight, dev_vB, matrixHeight);
    cudaDeviceSynchronize();

    cudaMemcpy(mX, dev_vB, sizeof(double) * matrixHeight * nrhs, cudaMemcpyDeviceToHost);

    // Timing Output
	cudaStreamSynchronize(operationStream);
	cublasDestroy(handle);
	cusolverDnDestroy(solver);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop); 
	cudaEventElapsedTime(&elapsedTimeInMiliseconds, start, stop); 

  	// Turned off to minimize printing
	printMatrix("Result X", mX, nrhs, matrixHeight);
	printf("\nStream and Event Time: %f Miliseconds\n", elapsedTimeInMiliseconds) * 100;

	// Free reserved memory
    free(mA); 
	free(vB);
	free(mX);
	cudaFree(dev_mA);
	cudaFree(dev_vB);
	cudaFree(dev_tau);
	cudaFree(dev_work);
	cudaFree(dev_Info);
	cublasShutdown();
	cudaDeviceReset();
}

int main(int argc, char** argv) {
	int matrixHeight = 1024;

	// read command line arguments
	if (argc >= 2) {
		matrixHeight = atoi(argv[1]);
	}

	printf("\nMatrix Width and Height: %d\n", matrixHeight);
	
	runOperation(matrixHeight, matrixHeight, 1);

	return 0;
}
