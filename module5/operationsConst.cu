#include <stdio.h>
#include <time.h>

// Initialize constant variables
__constant__ int* constThreadCountList;
__constant__ int* constRandNumList;
__constant__ int* constAddresultList;
__constant__ int* constSubresultList;
__constant__ int* constMultresultList;
__constant__ int* constModresultList;

/**
* addConstCuda: A method that add two arrays and places the result in a third array using 
* multithreading for index calculation using constant memory.
*/
__global__ void addConstCUDA() { 
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	constAddresultList[idx] = constThreadCountList[idx] + constRandNumList[idx]; 
}

/**
* subConstCuda: A method that substract two arrays and places the result in a third array using 
* multithreading for index calculation using constant memory.
*/
__global__ void subConstCUDA() { 
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	constSubresultList[idx] = constThreadCountList[idx] - constRandNumList[idx]; 
}

/**
* multConstCuda: A method that multiplies two arrays and places the result in a third array using 
* multithreading for index calculation using constant memory.
*/
__global__ void multConstCUDA() { 
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	constMultresultList[idx] = constThreadCountList[idx] * constRandNumList[idx]; 
}

/**
* modConstCuda: A method that does the modulus between two arrays and places the result in a third 
* array using multithreading for index calculation using constant memory.
*/
__global__ void modConstCUDA() { 
	const int idx = threadIdx.x + (blockIdx.x * blockDim.x); 
	constModresultList[idx] = constThreadCountList[idx] % constRandNumList[idx]; 
}

/**
* printArray: A method that takes in an a label and an array with its size and it feeds it to printf.
*/
void printArray(const char* name, const int *array, const int size) {
	printf("\n%s: [ ", name);
	for(int idx = 0; idx < size; idx++) {
		printf("%i ", array[idx]);
	}
	printf("]\n");
}

/**
* runOperations: Taking the number of blocks and threads it does 4 operations on the two 
* given arrays and prints their results. Uses const memory
*/
void runOperations(const int numBlocks, const int totalThreads, const int *threadCountList, const int *randNumList) { 
	
	// Set up input constant variables
	cudaMemcpyToSymbol(constThreadCountList, &threadCountList, sizeof(int) * totalThreads, 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(constRandNumList, &randNumList, sizeof(int) * totalThreads, 0, cudaMemcpyHostToDevice);

	// Prepare result array variables
	int* addresultList = (int*) malloc(totalThreads * sizeof(int));
	int* subresultList = (int*) malloc(totalThreads * sizeof(int));
	int* multresultList = (int*) malloc(totalThreads * sizeof(int));
	int* modresultList = (int*) malloc(totalThreads * sizeof(int));

	// Execute each operation and bring result from device to host
	addConstCUDA<<<numBlocks,totalThreads>>> ();
	cudaDeviceSynchronize();
	cudaMemcpyToSymbol(constAddresultList, &addresultList, sizeof(int) * totalThreads, 0, cudaMemcpyDeviceToHost);

	subConstCUDA<<<numBlocks,totalThreads>>> ();
	cudaDeviceSynchronize();
	cudaMemcpyToSymbol(constSubresultList, &subresultList, sizeof(int) * totalThreads, 0, cudaMemcpyDeviceToHost);

	multConstCUDA<<<numBlocks,totalThreads>>> ();
	cudaDeviceSynchronize();
	cudaMemcpyToSymbol(constMultresultList, &multresultList, sizeof(int) * totalThreads, 0, cudaMemcpyDeviceToHost);

	modConstCUDA<<<numBlocks,totalThreads>>> ();
	cudaDeviceSynchronize();
	cudaMemcpyToSymbol(constModresultList, &modresultList, sizeof(int) * totalThreads, 0, cudaMemcpyDeviceToHost);

	// Turned of to minimize printing
	printArray("Add Result", addresultList, totalThreads);
	printArray("Sub Result", subresultList, totalThreads);
	printArray("Mult Result", multresultList, totalThreads);
	printArray("Mod Result", modresultList, totalThreads);
}

/**
* runTest: Used to set up variables needed to run a timing test.
*/ 
void runTest(const int numBlocks, const int totalThreads) {

	// Set up variables for timing
	clock_t start, end;
	double timePassedMiliSeconds;

	// Set up global memory space 
	int* threadCountList = (int*) malloc(totalThreads * sizeof(int));
	int* randNumList = (int*) malloc(totalThreads * sizeof(int));
	
	// Populate paged memory arrays
	for ( int idx = 0; idx < totalThreads; idx++ ) {
    	threadCountList[idx] = idx; 
		randNumList[idx] = rand() % 4;
   	}
	
	// Show generated values
	// Turned of to minimize printing
	// printArray("Thread Count List", threadCountList, constTotalThreads);
	// printArray("Random Number List", randNumList, constTotalThreads);
	
	// Run and time operations using const memory
	start = clock();
	runOperations(numBlocks, totalThreads, threadCountList, randNumList);
	end = clock();
	timePassedMiliSeconds = (double) (end - start) * 1000 / CLOCKS_PER_SEC;
	printf("\nConstant Memory Time: %f Miliseconds\n", timePassedMiliSeconds);

	// Free device memory
	cudaDeviceReset();
}

/**
* Main method: starts the execution.
*/
int main(int argc, char** argv) {
	// Based on the work of Andrew Krepps
	
	// Set default values in case arguments don't come in command line.
	int totalThreads = 1024;
	int blockSize = 256;

	// read command line arguments
	if (argc >= 2) {
		totalThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = totalThreads/blockSize;

	// validate command line arguments
	if (totalThreads % blockSize != 0) {
		++numBlocks;
		totalThreads = numBlocks*blockSize;
		
		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", totalThreads);
	}

	printf("Total Threads: %d\nBlock Size: %d\n", totalThreads, blockSize);

	runTest(numBlocks, totalThreads);

	return 0;
}